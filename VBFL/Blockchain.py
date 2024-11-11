import copy

from Block import Block



class Blockchain:
    def __init__(self):
        self.chain = []
    
    def return_chain_structure(self):
        return self.chain
    
    def return_chain_length(self):
        return len(self.chain)
    
    def return_last_block(self):
        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            # blockchain doesn't even have its genesis block
            return None
    
    def return_last_block_pow_proof(self):
        if len(self.chain) > 0:
            return self.return_last_block().compute_hash(hash_entire_block= True)
        else:
            return None
    
    def replace_chain(self, chain):
        self.chain = copy.copy(chain)
    
    def append_block(self, block):
        self.chain.append(copy.copy(block))