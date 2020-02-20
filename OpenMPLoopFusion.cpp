#include "/home/amalik/LLVM/llvm-8.0.0.src/include/llvm/Pass.h"
#include "/home/amalik/LLVM/llvm-8.0.0.src/include/llvm/IR/Function.h"
#include "/home/amalik/LLVM/llvm-8.0.0.src/include/llvm/Support/raw_ostream.h"
#include "/home/amalik/LLVM/llvm-8.0.0.src/include/llvm/IR/Instructions.h"
#include "/home/amalik/LLVM/llvm-8.0.0.src/include/llvm/Analysis/LoopInfo.h"

using namespace std;
using namespace llvm;

#define DEBUG_TYPE "openmp-opt"



namespace{
	struct OpenMPLoopFusionPass: public FunctionPass{
		private:
			map <CallInst*, CallInst*> call_init_fini_mapping; 
			map <CallInst*, vector<CallInst*>> call_map;
			map <Value*, Value*> store_op0_op1;
			map<Value*, Value*> args_map;
			CallInst* current_call_init_instruction;
		public:
			static char ID;
			OpenMPLoopFusionPass() : FunctionPass(ID) {}
			virtual bool runOnFunction( Function &F ) override;
			void runOverTheBlocks( Function &F) ;
			bool check_the_compatibility();
			void replace_UseValues(Function &F);
			void cleanInstructions();
			void testPrinting();	
			void printFunction( Function &F);
	};
}


bool find(CallInst* I, map<CallInst*, vector<CallInst*>> m){

	if (m.size() == 0)
		return false;
	for ( auto itr = m.begin(); itr!=m.end(); ++itr){
		if ((itr->second).size()==0)
			continue;
		else{
			for ( auto itr1=(itr->second).begin(); itr1!=(itr->second).end(); ++itr1){
				if (I == *itr1)
					return true;
			}	
		}
	}
	return false;
}

bool OpenMPLoopFusionPass:: runOnFunction(Function &F){
// It will run over a Function as a basic block
//
		runOverTheBlocks(F);
		check_the_compatibility();
		cleanInstructions();
		replace_UseValues(F);
		//cleanInstructions();
		//testPrinting();
		printFunction(F);
	return false; // We did not change anything in the IR
}



void OpenMPLoopFusionPass::printFunction( Function &F ) {
		errs() <<"\nThis is the new IR for the fused stuff\n\n";
		F.print(errs(), nullptr);
}

void OpenMPLoopFusionPass::cleanInstructions(){
	for ( auto itr = call_map.begin(); itr != call_map.end(); itr++){
		Instruction *I = call_init_fini_mapping.find(itr->first)->second;
		int count = (itr->second).size();
		if (count == 0) 
			continue;
		else
			I->eraseFromParent();

		for (auto itr1 = (itr->second).begin(); itr1 != (itr->second).end(); itr1++){
			Instruction *I1 = *itr1;
		        I1->eraseFromParent();
			for ( auto itr2 = call_init_fini_mapping.begin(); itr2 != call_init_fini_mapping.end(); itr2++){
				if (*itr1 == itr2->first) {
                			Instruction *I2 = itr2->second;
					errs() << *I2 << "----:----" << count << "\n";
                			if (count == 1 || count ==0)
						break;
					else{
						I2->eraseFromParent();
						count--;
					}
				}
        		}

		}
	}
}


void OpenMPLoopFusionPass::replace_UseValues(Function &F){
	vector <Instruction*> remove;

	for (auto itr = call_map.begin(); itr != call_map.end(); itr++){
		vector<Value*> vm;
		for (Value *arg: (itr->first)->args()){
			vm.push_back(arg);
			}
		for (auto itr1 = (itr->second).begin(); itr1 != (itr->second).end(); itr1++){
			vector<Value*> vs;
			for (Value *arg: (*itr1)->args()){
				vs.push_back(arg);
			}

			for (auto vmitr=vm.begin(), vsitr=vs.begin(); vmitr!=vm.end() && vsitr!=vs.end(); vmitr++, vsitr++){
				if (isa<Constant>(*vmitr)) continue;
				 for (auto &BB : F) {
       				         for (auto &II : BB) {
						Instruction* It = &II;
						if (isa<CallInst>(It)) continue;
						for ( int k =0; k < It->getNumOperands(); k++){
							if (It->getOperand(k) == *vsitr){
								It->setOperand(k,*vmitr);
								if (isa<StoreInst>(It) && k > 0){
									remove.push_back(It);
								}
							}
						}
					 }

				 }
	
			}
		}
	}
	for ( auto r=remove.begin(); r!=remove.end(); r++)
		(*r)->eraseFromParent();
}


void OpenMPLoopFusionPass::runOverTheBlocks( Function &F){
	for (auto &BB : F) {
                for (auto &II : BB) {
			if (CallInst *c = dyn_cast<CallInst> (&II)){
					if (c->getCalledFunction()->getName() == "__kmpc_for_static_init_4"){
						current_call_init_instruction = c;
						}
					if (c->getCalledFunction()->getName() == "__kmpc_for_static_fini"){
						call_init_fini_mapping.insert({current_call_init_instruction, c});
						}
					}
			if (StoreInst *store = dyn_cast<StoreInst>(&II)){
				store_op0_op1.insert({store->getOperand(1),store->getOperand(0)});
				}
                	}
		}
}

bool OpenMPLoopFusionPass::check_the_compatibility(){
// Checking the compatibility of the two call instructions
//
	bool compatible = true;
	for (auto itr = call_init_fini_mapping.begin(); itr!=call_init_fini_mapping.end(); ++itr){
		vector <CallInst*> v;
		if (find(itr->first, call_map)) continue;
		for (auto itr1 = itr; itr1!=call_init_fini_mapping.end(); ++itr1){
			if (itr == itr1) continue;
			else {
					vector<Value*> v1, v2;
					for (Value* arg : (itr->first)->args()){
						v1.push_back(arg);
					}
					for (Value* arg2 : (itr1->first)->args()){
						v2.push_back(arg2);
					}
					for (auto i = v1.begin(), j=v2.begin(); i!=v1.end() && j!=v2.end(); ++i, ++j)
					{       

						if (isa<Constant>(*i) && isa<Constant>(*j))	{
							if (*i != *j){
								compatible=false; break;
							}
						}
						else{ // we have a pointer argument
							if (store_op0_op1.find(*j)->second != store_op0_op1.find(*i)->second){
								compatible=false; break;
							}	
						}
					}
					if (compatible){
						v.push_back(itr1->first);
					}
					else break;
			}
		}
		call_map.insert({itr->first, v});
		if (!compatible) {compatible=true;}
	}

	return true;
}


void OpenMPLoopFusionPass::testPrinting(){
	for ( auto itr = call_map.begin(); itr!=call_map.end(); ++itr){
                if ((itr->second).size()==0)
                        continue;
                else{
                        for ( auto itr1=(itr->second).begin(); itr1!=(itr->second).end(); ++itr1){
                                errs() << *(*itr1) << "--:---->\n[Main Instruction]" << *(itr->first) << "[testing]\n";
                        }
                }
        }
}

char OpenMPLoopFusionPass::ID = 0;

static RegisterPass<OpenMPLoopFusionPass> X("omplooppass", "Example LLVM pass printing each function it visits");


