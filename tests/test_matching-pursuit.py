from __future__ import print_function
import numpy as np

import MPnumpy as mpn
import MPnumbaprog as mpg

def setup__module():
    pass

def teardown_module():
    pass

class test_mp():

    def setup(self):
        self.rng = np.random.RandomState(0)
        self.numDict = 4096
        self.numBatch = 64
        self.dataSize = 64

    def teardown(self):
        pass

    def test_numpy(self):
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = (np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T))))).dot(dictionary)
        stimuli = self.rng.randn(self.numBatch,self.dataSize)
        c = mpn.mp(dictionary,stimuli)
        assert np.allclose(c.dot(dictionary),stimuli,atol=1e-5)

    def test_numbaprog(self):
        dictionary = self.rng.randn(self.numDict,self.dataSize)
        dictionary = np.array((np.sqrt(np.diag(1/np.diag(dictionary.dot(dictionary.T))))).dot(dictionary),dtype=np.float32,order='F')
        stimuli = np.array(self.rng.randn(self.numBatch,self.dataSize),dtype=np.float32,order='F')
        c = mpg.mp(dictionary,stimuli)
        assert np.allclose(c.dot(dictionary),stimuli,atol=1e-5)
