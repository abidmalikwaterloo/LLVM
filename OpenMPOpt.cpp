//===-- IPO/OpenMPOpt.cpp - Collection of OpenMP specific optimizations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP specific optimizations:
//
// - Deduplication of runtime calls, e.g., omp_get_thread_num.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/OpenMPOpt.h"

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CFG.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include <algorithm>

using namespace llvm;
using namespace omp;
using namespace types;

#define DEBUG_TYPE "openmp-opt"

static cl::opt<bool> DisableOpenMPOptimizations(
    "openmp-opt-disable", cl::ZeroOrMore,
    cl::desc("Disable OpenMP specific optimizations."), cl::Hidden,
    cl::init(false));

STATISTIC(NumOpenMPRuntimeCallsDeduplicated,
          "Number of OpenMP runtime calls deduplicated");
STATISTIC(NumOpenMPRuntimeFunctionsIdentified,
          "Number of OpenMP runtime functions identified");
STATISTIC(NumOpenMPRuntimeFunctionUsesIdentified,
          "Number of OpenMP runtime function uses identified");

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static constexpr auto TAG = "[" DEBUG_TYPE "]";
#endif

namespace {
struct OpenMPOpt {

  OpenMPOpt(SmallPtrSetImpl<Function *> &SCC,
            SmallPtrSetImpl<Function *> &ModuleSlice,
            CallGraphUpdater &CGUpdater)
      : M(*(*SCC.begin())->getParent()), SCC(SCC), ModuleSlice(ModuleSlice),
        OMPBuilder(M), CGUpdater(CGUpdater) {
    initializeTypes(M);
    initializeRuntimeFunctions();
    OMPBuilder.initialize();
  }

  /// Data structure to hold information for the  deleting
  /// redundent OpenMP for loop calls
  struct OMPLoopFusion {
    bool check=false;
    /// Keeps map of __kmpc_static_init4 and its __kmpc_static_fini calls for each OpenMP for loop
    std::map<CallInst *, CallInst *> call_init_fini_mapping;
    std::map<CallInst *, BasicBlock*> call_basicblock_mapping;
    /// Keeps map of __kmpc_static_init4 and  all its compatilable __kmpc_static_init4 in a vector
    std::map<CallInst *, std::vector<CallInst *>> call_map;
    std::map<CallInst *, std::vector<Value *>> call_arg;
    /// the data structure maintain the basic blocks in a lineage
    std::map<BasicBlock*, std::vector<BasicBlock*>> chain;
    std::vector<BasicBlock*> visited, loopVisited;
    /// store_op0_op01 keeps map of operand 1 and operand 0
    /// args_map keeps map of arguments of __kmpc_static_init4 for later cleaning
    std::map<Value *, Value *> store_op0_op1, args_map;
    CallInst *current_call_init_instruction = nullptr;
  };

  /// Generic information that describes a runtime function
  struct RuntimeFunctionInfo {
    /// The kind, as described by the RuntimeFunction enum.
    RuntimeFunction Kind;

    /// The name of the function.
    StringRef Name;

    /// Flag to indicate a variadic function.
    bool IsVarArg;

    /// The return type of the function.
    Type *ReturnType;

    /// The argument types of the function.
    SmallVector<Type *, 8> ArgumentTypes;

    /// The declaration if available.
    Function *Declaration;

    /// Uses of this runtime function per function containing the use.
    DenseMap<Function *, SmallPtrSet<Use *, 16>> UsesMap;

    /// Return the number of arguments (or the minimal number for variadic
    /// functions).
    size_t getNumArgs() const { return ArgumentTypes.size(); }

    /// Run the callback \p CB on each use and forget the use if the result is
    /// true. The callback will be fed the function in which the use was
    /// encountered as second argument.
    void foreachUse(function_ref<bool(Use &, Function &)> CB) {
      SmallVector<Use *, 8> ToBeDeleted;
      for (auto &It : UsesMap) {
        ToBeDeleted.clear();
        for (Use *U : It.second)
          if (CB(*U, *It.first))
            ToBeDeleted.push_back(U);
        for (Use *U : ToBeDeleted)
          It.second.erase(U);
      }
    }
  };

  /// Run all OpenMP optimizations on the underlying SCC/ModuleSlice.
  bool run() {
    bool Changed = false;
    LLVM_DEBUG(dbgs() << TAG << "Run on SCC with " << SCC.size()
                      << " functions in a slice with " << ModuleSlice.size()
                      << " functions\n");

    Changed |= deduplicateRuntimeCalls();
    Changed |= deleteParallelRegions();
    Changed |= deleteStaticScheduleCalls();

    return Changed;
  }

private:
  /// Combine "OpenMP for loop with static scheduling"
  /// check if all parameters are same and the loops are adjacent
  /// See https://openmp.llvm.org/Reference.pdf. See section 5.8.3.24 for parameters
  /// The two for loops can share the same __kmpc_static_init4() and __kmpc_static_fini()
  /// calls.
    
  bool deleteStaticScheduleCalls() {
    bool Changed = false;
    // if there is no kmpc_for_static_init_4, there is  no need to do anything
    RuntimeFunctionInfo &RFI = RFIs[OMPRTL___kmpc_for_static_init_4];
    if (!RFI.Declaration)
        return Changed;
    // Else go through each function
    OMPLoopFusion OLF;
    for (Function *F : SCC)
        Changed = runOverTheBlock(*F, &OLF);
    return Changed;
  }
    
// Check the compatility of the of the __kmpc_for_static_init_4
    void checkTheCompatibility(OMPLoopFusion *OLF){
        bool compatible = true;
        for (auto itr : OLF->call_init_fini_mapping) {
          if (find(itr.first, OLF->call_map)) continue;
          std::vector<CallInst *> v;
          std::vector<Value *> v1;
          for (Value *arg : (itr.first)->args())
                v1.push_back(arg);
          for (auto itr1 : OLF->call_init_fini_mapping) {
            if ((itr.first) == (itr1.first)) continue;
            if (find(itr1.first, OLF->call_map)) continue;
            std::vector<Value *> v2;
            for (Value *arg2 : (itr1.first)->args())
                v2.push_back(arg2);
            for (auto i = v1.begin(), j = v2.begin(); i != v1.end() && j != v2.end(); ++i, ++j) {
                if (isa<Constant>(*i) && isa<Constant>(*j)) {
                  if (*i != *j) {compatible = false; break;}
                }
                else {
                  if (OLF->store_op0_op1.find(*j)->second != OLF->store_op0_op1.find(*i)->second) {
                    compatible = false; break;}
                }
              }
            if (compatible) {
                for (auto i = v1.begin(), j = v2.begin(); i != v1.end() && j != v2.end(); ++i, ++j) {
                    OLF->args_map.insert({*j,*i});
                }
                v.push_back(itr1.first);
              }
              else break; /// the adjacent for omp loop is not compatible so there is no need to check others
               ///  therefore we need to break out of the second for loop
          }
            /// if a call instruction has some compatible call instructions then put in the call_map container
             OLF->call_map.insert({itr.first, v});
            ///  make the flag  true again for the next  instruction checking
            if (!compatible) compatible = true;
            v.clear();
        }
    }
    
    bool checkForOMPInit(BasicBlock*  B){
        if (!B) return false;
        for (BasicBlock::iterator BBI=B->begin(); BBI !=B->end(); ++BBI){
            if (CallInst *c= dyn_cast<CallInst>(BBI)){
                if (c->getCalledFunction()->getName()=="__kmpc_for_static_init_4"){
                    return true;}
                }
            }
        return false;
    }
    
    bool checkForOMPFini(BasicBlock*  B){
        if  (!B) return false;
        for (BasicBlock::iterator BBI=B->begin(); BBI !=B->end(); ++BBI){
            if (CallInst *c= dyn_cast<CallInst>(BBI)){
                if (c->getCalledFunction()->getName()=="__kmpc_for_static_fini"){
                    return true;}
                }
            }
        return false;
    }
    
    void markNodeVisited(BasicBlock* B,std::vector <BasicBlock *> &v,OMPLoopFusion *OLF){
        if (!B) return;
        OLF->visited.push_back(B);
        v.push_back(B);
        for ( auto BB: successors(B)){
            if (find(OLF->visited,BB)) continue;
            markNodeVisited(BB,v, OLF);
        }
    }
    
    BasicBlock* checkTheLoop(BasicBlock* B,std::vector <BasicBlock *> &v, OMPLoopFusion *OLF){
        std::vector<BasicBlock*> v2;
        for (auto S: successors(B)){
            if (checkLoop(S, B, v2)) {
                // mark all the node as visited
                markNodeVisited(S,v,OLF);
                return  nullptr;}
            else
                return S;
        }
        return nullptr;
    }
    
    bool checkLoop(BasicBlock* S, BasicBlock* B, std::vector<BasicBlock*>& visit){
        bool loop = false;
        if (!S) return loop;
        for (auto BB: successors(S)){
            if (BB == B) {loop = true; break;}
            if (find(visit, BB)) continue;
            visit.push_back(BB);
            loop = (loop || checkLoop (BB, B, visit));
        }
        return loop;
    }
    
    
    int countSuccessors(BasicBlock* B){
        int count = 0;
        for (auto BS: successors(B)) // I should use iterator instead
            count++;
        return count;
    }
    int countPredessors(BasicBlock* B){
        int count = 0;
        for (auto BP: predecessors(B))
            count++;
        return count;
    }
    void makeLineage(BasicBlock *B, std::vector <BasicBlock *> &v, OMPLoopFusion *OLF){
        if (!B or find(OLF->visited, B) ) return;
        if ((countSuccessors(B) <=1 )   && (countPredessors(B) > 1)) return; // unique entrance with two control flows
        if ((countPredessors(B) <=1 ) && (countSuccessors(B)) > 1) return; // two control flows merging into a unique point
        // these points can not  be part  of lineage for the optimizations
        BasicBlock* t=nullptr;
        // If you  have a basic blokc try to find the omp for starting point
        if (B->getSingleSuccessor()){
            OLF->visited.push_back(B);
            v.push_back(B);
            if  (checkForOMPInit(B)) // if you find it then find the end points ; all inbetween points are are part of the lineage
                t=checkOMPForLoop(B->getSingleSuccessor(), v, OLF);// the output is the basicblock for building the lineage
            else
                t=B->getSingleSuccessor();}// else take the  successor and move on
        else {// if you have a codition with more than two successors and predecessors
            // we need to check if they are control points or inbetween for loops
                OLF->visited.push_back(B);
                t = checkTheLoop(B, v, OLF) ;
                v.push_back(B);
        }
        makeLineage(t, v, OLF);
        return;
    }

    
    BasicBlock* checkOMPForLoop(BasicBlock *BB,std::vector <BasicBlock *> &v, OMPLoopFusion *OLF){
        BasicBlock * t = nullptr;
        if (!BB) return t;
        OLF->visited.push_back(BB);
        v.push_back(BB);
        for (auto B: successors(BB)){
           if (find(OLF->visited, B)) continue;
           if (checkForOMPFini(B)) { t= B; continue;}
           checkOMPForLoop (B, v, OLF);
        }
        return t;
    }
   
    
    bool find(std::vector <BasicBlock*> b, BasicBlock* B){
        for ( auto t: b)
            if (t == B)  return true;
        return false;
    }
    
    bool find(CallInst *I, std::map<CallInst *, std::vector<CallInst *>> m) {
        for (auto itr :m){
            if (itr.first== I) return true;
         for (auto itr1 : (itr.second))
             if (I == itr1) return true;
        }
       return false;
    }
    
    void clean_intrinsic_calls(BasicBlock* B, OMPLoopFusion *OLF){
        std::vector<Instruction *> remove;
        for (BasicBlock::iterator DI = B->begin(); DI != B->end(); ++DI ) {
            if (IntrinsicInst *II = dyn_cast<IntrinsicInst> (DI)){
                if (II->getIntrinsicID() == Intrinsic::lifetime_start || II->getIntrinsicID() == Intrinsic::lifetime_end ){
                    remove.push_back(II);
                    }
                }
            }
        for (auto r: remove)
            r->eraseFromParent();
    }
    
    void check_call_instructions(BasicBlock* B, OMPLoopFusion *OLF){
        for (BasicBlock::iterator DI = B->begin(); DI != B->end(); ++DI ) {
            if (CallInst *c = dyn_cast<CallInst>(DI)) {
              if (c->getCalledFunction()->getName() == "__kmpc_for_static_init_4")
                OLF->current_call_init_instruction = c;
              if (c->getCalledFunction()->getName() == "__kmpc_for_static_fini")
                OLF->call_init_fini_mapping.insert({OLF->current_call_init_instruction, c});
            }
            if (StoreInst *store = dyn_cast<StoreInst>(DI))
                OLF->store_op0_op1.insert({store->getOperand(1), store->getOperand(0)});
          }
    }
                                 
    bool runOverTheBlock(Function &F, OMPLoopFusion *OLF) {
        std::vector <BasicBlock *> v;
        bool changed = false;
         for (auto &BB: F) {
             // on each  block prepare data structure for the instructions
             if (find (OLF->visited, &BB)) continue;
             makeLineage (&BB, v, OLF);
             OLF->chain.insert({&BB,v});
             v.clear();
         }
        changed = doTheOptimization(OLF);// act  on the formed lineages
        
        return changed;
    }
     
    bool doTheOptimization(OMPLoopFusion *OLF){
        bool changed = false;
        for (auto S: OLF->chain){
            //we have todo it for each lineage
            //B is a basic block in a lineage
            for ( auto B:S.second){
                check_call_instructions(B, OLF);
            }
            checkTheCompatibility(OLF);
            changed = cleanInstructions(OLF);
            if (changed)
                for (auto B:S.second){
                    replace_UseValues(B, OLF);
                    clean_intrinsic_calls(B, OLF);
                }
            OLF->call_init_fini_mapping.clear();
            OLF->call_map.clear();
            OLF->store_op0_op1.clear();
            OLF->args_map.clear();
            
        }
        return changed;
    }
    
    void replace_UseValues(BasicBlock* B, OMPLoopFusion *OLF){
        std::vector<Instruction *> remove;
        for (BasicBlock::iterator II = B->begin(); II != B->end(); ++II) {
            Instruction *It = dyn_cast<Instruction>(II);
            if (isa<CallInst>(It)) continue;
            for (unsigned int k = 0; k < It->getNumOperands(); k++){
                auto temp =  OLF->args_map.find(It->getOperand(k));
                if (temp != OLF->args_map.end()){
                    It->setOperand(k, temp->second);
                    if (isa<StoreInst>(It) && k > 0) remove.push_back(It);
                    }
            }
        }
        for (auto r: remove)
                r->eraseFromParent();
    }
    
    bool cleanInstructions(OMPLoopFusion *OLF) {
      bool changed = false;
      for (auto itr : OLF->call_map) {
        int count = (itr.second).size();
        if (!count) continue;
        Instruction *I = OLF->call_init_fini_mapping.find(itr.first)->second;
        I->eraseFromParent();
        changed = true;
        for (auto itr1:itr.second) {
          Instruction *I1 = itr1;
          Instruction *I2 = OLF->call_init_fini_mapping.find(itr1)->second;
          I1->eraseFromParent();
          if (count == 1) break;
          I2->eraseFromParent();
          count--;
          }
        }
      return changed;
    }
  

/// does function printing
void printFunction(Function &F) {
    F.print(errs(), nullptr);
  }

  /// Try to delete parallel regions if possible
  bool deleteParallelRegions() {
    const unsigned CallbackCalleeOperand = 2;

    RuntimeFunctionInfo &RFI = RFIs[OMPRTL___kmpc_fork_call];
    if (!RFI.Declaration)
      return false;

    bool Changed = false;
    auto DeleteCallCB = [&](Use &U, Function &) {
      CallInst *CI = getCallIfRegularCall(U);
      if (!CI)
        return false;
      auto *Fn = dyn_cast<Function>(
          CI->getArgOperand(CallbackCalleeOperand)->stripPointerCasts());
      if (!Fn)
        return false;
      if (!Fn->onlyReadsMemory())
        return false;
      if (!Fn->hasFnAttribute(Attribute::WillReturn))
        return false;

      LLVM_DEBUG(dbgs() << TAG << "Delete read-only parallel region in "
                        << CI->getCaller()->getName() << "\n");
      CGUpdater.removeCallSite(*CI);
      CI->eraseFromParent();
      Changed = true;
      return true;
    };

    RFI.foreachUse(DeleteCallCB);

    return Changed;
  }

  /// Try to eliminiate runtime calls by reusing existing ones.
  bool deduplicateRuntimeCalls() {
    bool Changed = false;

    RuntimeFunction DeduplicableRuntimeCallIDs[] = {
        OMPRTL_omp_get_num_threads,
        OMPRTL_omp_in_parallel,
        OMPRTL_omp_get_cancellation,
        OMPRTL_omp_get_thread_limit,
        OMPRTL_omp_get_supported_active_levels,
        OMPRTL_omp_get_level,
        OMPRTL_omp_get_ancestor_thread_num,
        OMPRTL_omp_get_team_size,
        OMPRTL_omp_get_active_level,
        OMPRTL_omp_in_final,
        OMPRTL_omp_get_proc_bind,
        OMPRTL_omp_get_num_places,
        OMPRTL_omp_get_num_procs,
        OMPRTL_omp_get_place_num,
        OMPRTL_omp_get_partition_num_places,
        OMPRTL_omp_get_partition_place_nums};

    // Global-tid is handled separatly.
    SmallSetVector<Value *, 16> GTIdArgs;
    collectGlobalThreadIdArguments(GTIdArgs);
    LLVM_DEBUG(dbgs() << TAG << "Found " << GTIdArgs.size()
                      << " global thread ID arguments\n");

    for (Function *F : SCC) {
      for (auto DeduplicableRuntimeCallID : DeduplicableRuntimeCallIDs)
        deduplicateRuntimeCalls(*F, RFIs[DeduplicableRuntimeCallID]);

      // __kmpc_global_thread_num is special as we can replace it with an
      // argument in enough cases to make it worth trying.
      Value *GTIdArg = nullptr;
      for (Argument &Arg : F->args())
        if (GTIdArgs.count(&Arg)) {
          GTIdArg = &Arg;
          break;
        }
      Changed |= deduplicateRuntimeCalls(
          *F, RFIs[OMPRTL___kmpc_global_thread_num], GTIdArg);
    }

    return Changed;
  }

  /// Try to eliminiate calls of \p RFI in \p F by reusing an existing one or
  /// \p ReplVal if given.
  bool deduplicateRuntimeCalls(Function &F, RuntimeFunctionInfo &RFI,
                               Value *ReplVal = nullptr) {
    auto &Uses = RFI.UsesMap[&F];
    if (Uses.size() + (ReplVal != nullptr) < 2)
      return false;

    LLVM_DEBUG(dbgs() << TAG << "Deduplicate " << Uses.size() << " uses of "
                      << RFI.Name
                      << (ReplVal ? " with an existing value\n" : "\n")
                      << "\n");
    assert((!ReplVal || (isa<Argument>(ReplVal) &&
                         cast<Argument>(ReplVal)->getParent() == &F)) &&
           "Unexpected replacement value!");
    if (!ReplVal) {
      for (Use *U : Uses)
        if (CallInst *CI = getCallIfRegularCall(*U, &RFI)) {
          CI->moveBefore(&*F.getEntryBlock().getFirstInsertionPt());
          ReplVal = CI;
          break;
        }
      if (!ReplVal)
        return false;
    }

    bool Changed = false;
    auto ReplaceAndDeleteCB = [&](Use &U, Function &Caller) {
      CallInst *CI = getCallIfRegularCall(U, &RFI);
      if (!CI || CI == ReplVal || &F != &Caller)
        return false;
      assert(CI->getCaller() == &F && "Unexpected call!");
      CGUpdater.removeCallSite(*CI);
      CI->replaceAllUsesWith(ReplVal);
      CI->eraseFromParent();
      ++NumOpenMPRuntimeCallsDeduplicated;
      Changed = true;
      return true;
    };
    RFI.foreachUse(ReplaceAndDeleteCB);

    return Changed;
  }

  /// Collect arguments that represent the global thread id in \p GTIdArgs.
  void collectGlobalThreadIdArguments(SmallSetVector<Value *, 16> &GTIdArgs) {
    // TODO: Below we basically perform a fixpoint iteration with a pessimistic
    //       initialization. We could define an AbstractAttribute instead and
    //       run the Attributor here once it can be run as an SCC pass.

    // Helper to check the argument \p ArgNo at all call sites of \p F for
    // a GTId.
    auto CallArgOpIsGTId = [&](Function &F, unsigned ArgNo, CallInst &RefCI) {
      if (!F.hasLocalLinkage())
        return false;
      for (Use &U : F.uses()) {
        if (CallInst *CI = getCallIfRegularCall(U)) {
          Value *ArgOp = CI->getArgOperand(ArgNo);
          if (CI == &RefCI || GTIdArgs.count(ArgOp) ||
              getCallIfRegularCall(*ArgOp,
                                   &RFIs[OMPRTL___kmpc_global_thread_num]))
            continue;
        }
        return false;
      }
      return true;
    };

    // Helper to identify uses of a GTId as GTId arguments.
    auto AddUserArgs = [&](Value &GTId) {
      for (Use &U : GTId.uses())
        if (CallInst *CI = dyn_cast<CallInst>(U.getUser()))
          if (CI->isArgOperand(&U))
            if (Function *Callee = CI->getCalledFunction())
              if (CallArgOpIsGTId(*Callee, U.getOperandNo(), *CI))
                GTIdArgs.insert(Callee->getArg(U.getOperandNo()));
    };

    // The argument users of __kmpc_global_thread_num calls are GTIds.
    RuntimeFunctionInfo &GlobThreadNumRFI =
        RFIs[OMPRTL___kmpc_global_thread_num];
    for (auto &It : GlobThreadNumRFI.UsesMap)
      for (Use *U : It.second)
        if (CallInst *CI = getCallIfRegularCall(*U, &GlobThreadNumRFI))
          AddUserArgs(*CI);

    // Transitively search for more arguments by looking at the users of the
    // ones we know already. During the search the GTIdArgs vector is extended
    // so we cannot cache the size nor can we use a range based for.
    for (unsigned u = 0; u < GTIdArgs.size(); ++u)
      AddUserArgs(*GTIdArgs[u]);
  }

  /// Return the call if \p U is a callee use in a regular call. If \p RFI is
  /// given it has to be the callee or a nullptr is returned.
  CallInst *getCallIfRegularCall(Use &U, RuntimeFunctionInfo *RFI = nullptr) {
    CallInst *CI = dyn_cast<CallInst>(U.getUser());
    if (CI && CI->isCallee(&U) && !CI->hasOperandBundles() &&
        (!RFI || CI->getCalledFunction() == RFI->Declaration))
      return CI;
    return nullptr;
  }

  /// Return the call if \p V is a regular call. If \p RFI is given it has to be
  /// the callee or a nullptr is returned.
  CallInst *getCallIfRegularCall(Value &V, RuntimeFunctionInfo *RFI = nullptr) {
    CallInst *CI = dyn_cast<CallInst>(&V);
    if (CI && !CI->hasOperandBundles() &&
        (!RFI || CI->getCalledFunction() == RFI->Declaration))
      return CI;
    return nullptr;
  }

  /// Helper to initialize all runtime function information for those defined in
  /// OpenMPKinds.def.
  void initializeRuntimeFunctions() {
    // Helper to collect all uses of the decleration in the UsesMap.
    auto CollectUses = [&](RuntimeFunctionInfo &RFI) {
      unsigned NumUses = 0;
      if (!RFI.Declaration)
        return NumUses;
      OMPBuilder.addAttributes(RFI.Kind, *RFI.Declaration);

      NumOpenMPRuntimeFunctionsIdentified += 1;
      NumOpenMPRuntimeFunctionUsesIdentified += RFI.Declaration->getNumUses();

      // TODO: We directly convert uses into proper calls and unknown uses.
      for (Use &U : RFI.Declaration->uses()) {
        if (Instruction *UserI = dyn_cast<Instruction>(U.getUser())) {
          if (ModuleSlice.count(UserI->getFunction())) {
            RFI.UsesMap[UserI->getFunction()].insert(&U);
            ++NumUses;
          }
        } else {
          RFI.UsesMap[nullptr].insert(&U);
          ++NumUses;
        }
      }
      return NumUses;
    };

#define OMP_RTL(_Enum, _Name, _IsVarArg, _ReturnType, ...)                     \
  {                                                                            \
    auto &RFI = RFIs[_Enum];                                                   \
    RFI.Kind = _Enum;                                                          \
    RFI.Name = _Name;                                                          \
    RFI.IsVarArg = _IsVarArg;                                                  \
    RFI.ReturnType = _ReturnType;                                              \
    RFI.ArgumentTypes = SmallVector<Type *, 8>({__VA_ARGS__});                 \
    RFI.Declaration = M.getFunction(_Name);                                    \
    unsigned NumUses = CollectUses(RFI);                                       \
    (void)NumUses;                                                             \
    LLVM_DEBUG({                                                               \
      dbgs() << TAG << RFI.Name << (RFI.Declaration ? "" : " not")             \
             << " found\n";                                                    \
      if (RFI.Declaration)                                                     \
        dbgs() << TAG << "-> got " << NumUses << " uses in "                   \
               << RFI.UsesMap.size() << " different functions.\n";             \
    });                                                                        \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"

    // TODO: We should validate the declaration agains the types we expect.
    // TODO: We should attach the attributes defined in OMPKinds.def.
  }

  /// The underyling module.
  Module &M;

  /// The SCC we are operating on.
  SmallPtrSetImpl<Function *> &SCC;

  /// The slice of the module we are allowed to look at.
  SmallPtrSetImpl<Function *> &ModuleSlice;

  /// An OpenMP-IR-Builder instance
  OpenMPIRBuilder OMPBuilder;

  /// Callback to update the call graph, the first argument is a removed call,
  /// the second an optional replacement call.
  CallGraphUpdater &CGUpdater;

  /// Map from runtime function kind to the runtime function description.
  EnumeratedArray<RuntimeFunctionInfo, RuntimeFunction,
                  RuntimeFunction::OMPRTL___last>
      RFIs;
};
} // namespace

PreservedAnalyses OpenMPOptPass::run(LazyCallGraph::SCC &C,
                                     CGSCCAnalysisManager &AM,
                                     LazyCallGraph &CG, CGSCCUpdateResult &UR) {
  if (!containsOpenMP(*C.begin()->getFunction().getParent(), OMPInModule))
    return PreservedAnalyses::all();

  if (DisableOpenMPOptimizations)
    return PreservedAnalyses::all();

  SmallPtrSet<Function *, 16> SCC;
  for (LazyCallGraph::Node &N : C)
    SCC.insert(&N.getFunction());

  if (SCC.empty())
    return PreservedAnalyses::all();

  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);
  // TODO: Compute the module slice we are allowed to look at.
  OpenMPOpt OMPOpt(SCC, SCC, CGUpdater);
  bool Changed = OMPOpt.run();
  (void)Changed;
  return PreservedAnalyses::all();
}

namespace {

struct OpenMPOptLegacyPass : public CallGraphSCCPass {
  CallGraphUpdater CGUpdater;
  OpenMPInModule OMPInModule;
  static char ID;

  OpenMPOptLegacyPass() : CallGraphSCCPass(ID) {
    initializeOpenMPOptLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool doInitialization(CallGraph &CG) override {
    // Disable the pass if there is no OpenMP (runtime call) in the module.
    containsOpenMP(CG.getModule(), OMPInModule);
    return false;
  }

  bool runOnSCC(CallGraphSCC &CGSCC) override {
    if (!containsOpenMP(CGSCC.getCallGraph().getModule(), OMPInModule))
      return false;
    if (DisableOpenMPOptimizations || skipSCC(CGSCC))
      return false;

    SmallPtrSet<Function *, 16> SCC;
    for (CallGraphNode *CGN : CGSCC)
      if (Function *Fn = CGN->getFunction())
        if (!Fn->isDeclaration())
          SCC.insert(Fn);

    if (SCC.empty())
      return false;

    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    CGUpdater.initialize(CG, CGSCC);

    // TODO: Compute the module slice we are allowed to look at.
    OpenMPOpt OMPOpt(SCC, SCC, CGUpdater);
    return OMPOpt.run();
  }

  bool doFinalization(CallGraph &CG) override { return CGUpdater.finalize(); }
};

} // end anonymous namespace

bool llvm::omp::containsOpenMP(Module &M, OpenMPInModule &OMPInModule) {
  if (OMPInModule.isKnown())
    return OMPInModule;

#define OMP_RTL(_Enum, _Name, ...)                                             \
  if (M.getFunction(_Name))                                                    \
    return OMPInModule = true;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  return OMPInModule = false;
}

char OpenMPOptLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(OpenMPOptLegacyPass, "openmpopt",
                      "OpenMP specific optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(OpenMPOptLegacyPass, "openmpopt",
                    "OpenMP specific optimizations", false, false)

Pass *llvm::createOpenMPOptLegacyPass() { return new OpenMPOptLegacyPass(); }
