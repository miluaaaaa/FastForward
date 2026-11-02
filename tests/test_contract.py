import csv, math, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from shunong_accel.api import response, validate_request

class Contracts(unittest.TestCase):
 def test_request(self):
  with TemporaryDirectory() as t:
   image=Path(t)/"input.png"; image.touch()
   validate_request(image,"Question",["A","B"])
   for choices in (["A"],["A", ""], ["A"]*9):
    with self.assertRaises(ValueError): validate_request(image,"Question",choices)
 def test_response(self):
  r=response([1.,1.],["a","b"],"x")
  self.assertEqual(r["answer"]["letter"],"A")
  self.assertAlmostEqual(sum(r["probabilities"]),1.)
  with self.assertRaises(RuntimeError): response([float("nan"),0.],["a","b"],"x")
 def test_matrix(self):
  p=Path(__file__).resolve().parents[1]/"results/qwen-main.csv"
  with p.open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),192)
  self.assertEqual(len({(r["model"],r["benchmark"],r["method"]) for r in rows}),192)
  for r in rows:
   self.assertTrue(math.isfinite(float(r["mean_ms"])))
   self.assertGreater(float(r["mean_ms"]),0.)
