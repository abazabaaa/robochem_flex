"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR: Test the connection and operation of the RamaBerry device while the connection to the server is active
"""

import unittest
from tests.unit.test_device_Rama_Berry import RamaBerryTest
from omniplatypus.devices.nrg.rama_berry import RamaBerry


class TestActualConn(RamaBerryTest):
    """Same test as before, but with actual connection to the device, we need to manually
    start the server in the RamaBerry device and then run the tests"""

    # class setup:
    @classmethod
    def setUpClass(cls):
        """Setup runs before the test class."""
        cls.device = RamaBerry()
        cls.device.open("0.0.0.0", 12345)

    def setUp(self):
        """Setup runs before each test method."""
        pass

    def tearDown(self):
        """Teardown runs after each test method."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Teardown runs after the test class."""
        cls.device.close()


if __name__ == "__main__":
    Raman = RamaBerry()
    Raman.open("0.0.0.0", 99999)
    Raman['integration_time'] = 1000
    Raman['n_averages'] = 3
    Raman['save_as'] = "SERVER_SINGLE"
    Raman['save_moniker'] = 'test'

    Raman['start_acq'] = True

    data = Raman['data']

    print(Raman.usage())
    Raman.close()
