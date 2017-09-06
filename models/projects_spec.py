import unittest
from projects import Projects

def multiply(x,y):
   return x/y
 
class TestUM(unittest.TestCase):
 
    def setUp(self):
      self.p = Projects()
      pass
 
    def test_numbers_3_4(self):
      self.p.create({})
      self.assertEqual( multiply(3,4), 12)
 
    def test_strings_a_3(self):
        self.assertEqual( multiply('a',3), 'aaa')
 
if __name__ == '__main__':
    unittest.main()