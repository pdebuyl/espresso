from __future__ import print_function

import espressomd as md
import unittest as ut

@ut.skipIf(not md.has_features(['LEES_EDWARDS']),
  'Feature not available, skipping test!')

class LeesEdwardsInterfaceTest(ut.TestCase):

  def test(self):

    system = md.System(box_l=[10,10,10])
    
    system.lees_edwards = ["off"]
    self.assertListEqual(system.lees_edwards, ["off"])

    system.lees_edwards = ["step", 1.0]
    self.assertListEqual(system.lees_edwards, ["step", 1.0])
    
    system.lees_edwards = ["steady_shear", 0.1]
    self.assertListEqual(system.lees_edwards, ["steady_shear", 0.1])
    
    system.lees_edwards = ["oscillatory_shear", 0.5, 0.5]
    self.assertListEqual(system.lees_edwards, ["oscillatory_shear", 0.5, 0.5])

if __name__ == "__main__":
  ut.main()
