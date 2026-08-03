import os
import argparse
import torch
from torch_npu.testing.testcase import TestCase
from mindspore.ops import zeros_like

from torch.testing._internal.common_utils import seed_all


class TestAbs(TestCase):

    def test_api_completeness(self):
        self.test_abs_when_given_valid_input_then_return_expected_value()
        self.test_abs_when_given_valid_input_then_return_out_param_value()

    def test_abs_when_given_valid_input_then_return_expected_value(self):
        input_ = torch.tensor([-1, -2, 3])
        output_ = torch.abs(input_)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(output_, expected), f"for TestAbs, expected: {expected}, but got {output_}"

    def test_abs_when_given_valid_input_then_return_out_param_value(self):
        input_ = torch.tensor([-1, -2, 3])
        out = zeros_like(input_)
        torch.abs(input_, out=out)
        expected = torch.tensor([1, 2, 3])
        assert torch.equal(out, expected), f"for TestAbs out param, expected: {expected}, but got {out}"


class TestAdd(TestCase):

    def test_api_completeness(self):
        self.test_add_when_given_valid_input_then_return_expected_value()
        self.test_add_when_given_valid_input_then_return_out_param_value()
        self.test_add_when_given_valid_input_then_return_valid_precision()

    def test_add_when_given_valid_input_then_return_expected_value(self):
        input_1 = torch.tensor([0.7047168016, 0.6540585160, -0.3598807752, 0.0146041336])
        input_2 = torch.tensor([[0.6164773703],
                                [0.1437176913],
                                [-0.1017297357],
                                [-0.5562295318]])
        output_ = torch.add(input_1, input_2, alpha=10)  # auto broadcasting: a + b * alpha
        expected = torch.tensor([
            [6.86949062, 6.81883240, 5.80489302, 6.17937803],
            [2.14189386, 2.09123540, 1.07729614, 1.45178103],
            [-0.31258059, -0.36323887, -1.37717819, -1.00269330],
            [-4.85757875, -4.90823698, -5.92217636, -5.54769135]
        ])
        assert torch.allclose(output_, expected), f"for TestAdd, {expected}, but got {output_}"

    def test_add_when_given_valid_input_then_return_out_param_value(self):
        input_ = torch.tensor([16.801167, 18.46366, 19., 19.634846])
        out = torch.zeros_like(input_)
        torch.add(input_, 20, out=out)
        expected = torch.tensor([36.801167, 38.46366, 39., 39.634846])
        assert torch.allclose(out, expected), f"for TestAdd, expected: {expected}, but got {out}"

    def test_add_when_given_valid_input_then_return_valid_precision(self):
        input_ = torch.tensor([16.801167, 18.46366, 19., 19.634846])
        output_ = torch.add(input_, 20)
        print(output_)
        expected = torch.tensor([36.801167, 38.46366, 39., 39.634846])
        # percision err:
        assert torch.allclose(output_, expected), f"for TestAdd, expected: {expected}, but got {output_}"


class TestClamp(TestCase):

    def test_api_completeness(self):
        input_ = torch.tensor([2.3016286, 1.2361908, -0.1439718, 0.0350262])
        min_ = torch.linspace(-1, 1, steps=4)
        output_ = torch.clamp(input_, min=min_)
        print(output_)

        input_ = torch.tensor([-1.2858136, 0.4902603, 0.2593709, 0.7027668])
        max_ = torch.linspace(-1, 1, steps=4)
        output_ = torch.clamp(input_, max=max_)
        print(output_)

        input_ = torch.tensor([-2.1907523, 0.15, 0.6457138, 0.35])
        output_ = torch.clamp(input_, min=-0.5, max=0.5)
        print(output_)
        expected = torch.tensor([-0.5, 0.15, 0.5, 0.35])
        assert torch.allclose(output_, expected), f"for TestClamp, expected: {expected}, but got {output_}"


class TestCos(TestCase):

    def test_api_completeness(self):
        self.test_cos_when_given_valid_input_then_return_expected_value()
        self.test_cos_when_given_valid_input_then_return_out_param_value()

    def test_cos_when_given_valid_input_then_return_expected_value(self):
        input_ = torch.tensor([-0.2919261, 0.6896933, -0.8552471, 0.6651444])
        output_ = torch.cos(input_)
        expected = torch.tensor([0.9576913, 0.7714412, 0.6560320, 0.7868276])
        print(output_)
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_cos_when_given_valid_input_then_return_out_param_value(self):
        input_ = torch.tensor([-0.2919261, 0.6896933, -0.8552471, 0.6651444])
        out = zeros_like(input_)
        torch.cos(input_, out=out)
        expected = torch.tensor([0.9576913, 0.7714412, 0.6560320, 0.7868276])
        assert torch.allclose(out, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {out}"


class TestDiv(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()

    def test_api_completeness_1(self):
        input_1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                                [0.1815, -1.0111, 0.9805, -1.5923],
                                [0.1062, 1.4581, 0.7759, -1.2344],
                                [-0.1830, -0.0313, 1.1908, -1.4757]])
        input_2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        output_ = torch.div(input_1, input_2)
        print(output_)
        expected = torch.tensor([[-0.4620269, -6.6051192, 0.5676076, 1.2638649],
                                 [0.2259711, -3.4508533, -1.2085541, 6.8990469],
                                 [0.1322211, 4.9764500, -0.9563664, 5.3483539],
                                 [-0.2278386, -0.1068259, -1.4677677, 6.3938475]])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                                [0.1815, -1.0111, 0.9805, -1.5923],
                                [0.1062, 1.4581, 0.7759, -1.2344],
                                [-0.1830, -0.0313, 1.1908, -1.4757]])
        input_2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        output_ = torch.div(input_1, input_2, rounding_mode='floor')
        print(output_)
        expected = torch.tensor([[-1., -7., 0., 1.],
                                 [0., -4., -2., 6.],
                                 [0., 4., -1., 5.],
                                 [-1., -1., -2., 6.]])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_3(self):
        input_1 = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
                                [0.1815, -1.0111, 0.9805, -1.5923],
                                [0.1062, 1.4581, 0.7759, -1.2344],
                                [-0.1830, -0.0313, 1.1908, -1.4757]])
        input_2 = torch.tensor([0.8032, 0.2930, -0.8113, -0.2308])
        output_ = torch.div(input_1, input_2, rounding_mode='trunc')
        print(output_)
        expected = torch.tensor([[-0., -6., 0., 1.],
                                 [0., -3., -1., 6.],
                                 [0., 4., -0., 5.],
                                 [-0., -0., -1., 6.]])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestExp(TestCase):

    def test_api_completeness(self):
        input_ = torch.tensor([0, 1])
        output_ = torch.exp(input_)
        print(output_)
        expected = torch.tensor([1.0000000, 2.7182817])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestLog(TestCase):

    def test_api_completeness(self):
        input_ = torch.tensor([3.6958766, 3.9205103, 0.8964190, 1.5646839, 2.9774647])
        output_ = torch.log(input_)
        print(output_)
        expected = torch.tensor([1.3072177, 1.3662218, -0.1093474, 0.4476838, 1.0910722])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestLog2(TestCase):

    def test_api_completeness(self):
        input_ = torch.tensor([0.5300609, 0.5614995, 0.9711254, 0.6526233, 0.6825761])
        output_ = torch.log2(input_)
        print(output_)
        expected = torch.tensor([-0.9157700, -0.8326433, -0.0422705, -0.6156776, -0.5509381])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestLogicalAnd(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()

    def test_api_completeness_1(self):
        input_1 = torch.tensor([True, False, True])
        input_2 = torch.tensor([True, False, False])

        output_ = torch.logical_and(input_1, input_2)
        print(output_)
        expected = torch.tensor([True, False, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_1 = torch.tensor([0, 1, 10, 0], dtype=torch.bfloat16)
        input_2 = torch.tensor([4, 0, 1, 0], dtype=torch.int8)

        output_ = torch.logical_and(input_1, input_2)
        print(output_)
        expected = torch.tensor([False, False, True, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_3(self):
        input_1 = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        input_2 = torch.tensor([4, 0, 1, 0], dtype=torch.int8)

        output_ = torch.logical_and(input_1, input_2)
        print(output_)
        expected = torch.tensor([False, False, True, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestLogicalNot(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()
        self.test_api_completeness_5()

    def test_api_completeness_1(self):
        input_1 = torch.tensor([True, False])
        output_ = torch.logical_not(input_1)
        print(output_)
        expected = torch.tensor([False, True])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_1 = torch.tensor([0, 1, -10], dtype=torch.int8)
        output_ = torch.logical_not(input_1)
        print(output_)
        expected = torch.tensor([True, False, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_3(self):
        input_1 = torch.tensor([0., 1.5, -10.], dtype=torch.bfloat16)
        output_ = torch.logical_not(input_1)
        print(output_)
        expected = torch.tensor([True, False, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_4(self):
        input_1 = torch.tensor([0., 1.5, -10.], dtype=torch.float16)
        output_ = torch.logical_not(input_1)
        print(output_)
        expected = torch.tensor([True, False, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_5(self):
        input_1 = torch.tensor([0., 1.5, -10.], dtype=torch.float32)
        output_ = torch.logical_not(input_1)
        print(output_)
        expected = torch.tensor([True, False, False])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestLogicalOr(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()

    def test_api_completeness_1(self):
        input_1 = torch.tensor([True, False, True])
        input_2 = torch.tensor([True, False, False])
        output_ = torch.logical_or(input_1, input_2)
        print(output_)
        expected = torch.tensor([True, False, True])
        assert torch.allclose(output_, expected), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_1 = torch.tensor([0, 1, 10, 0], dtype=torch.bfloat16)
        input_2 = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        output_ = torch.logical_or(input_1, input_2)
        print(output_)
        expected = torch.tensor([True, True, True, False])
        assert torch.allclose(output_, expected), f"expected: {expected}, but got {output_}"

    def test_api_completeness_3(self):
        input_1 = torch.tensor([0, 1, 10, 0], dtype=torch.float16)
        input_2 = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        output_ = torch.logical_or(input_1, input_2)
        print(output_)
        expected = torch.tensor([True, True, True, False])
        assert torch.allclose(output_, expected), f"expected: {expected}, but got {output_}"

    def test_api_completeness_4(self):
        input_1 = torch.tensor([0, 1, 10, 0], dtype=torch.float32)
        input_2 = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
        output_ = torch.logical_or(input_1, input_2)
        print(output_)
        expected = torch.tensor([True, True, True, False])
        assert torch.allclose(output_, expected), f"expected: {expected}, but got {output_}"


class TestMul(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()

    def test_api_completeness_1(self):
        input_1 = torch.tensor([-1.9696293, 0.6183859, 0.8636860])
        output_ = torch.mul(input_1, 10)
        print(output_)
        expected = torch.tensor([-19.69629211, 6.18385925, 8.63685989])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_1 = torch.tensor([[0.2036932],
                                [-1.4942740],
                                [1.1691655],
                                [0.4778146]])
        input_2 = torch.tensor([[-0.0818304, 0.7099793, 0.1022753, 1.2349162]])
        output_ = torch.mul(input_1, input_2)
        print(output_)
        expected = torch.tensor([[-0.0166683, 0.1446179, 0.0208328, 0.2515440],
                                 [0.1222771, -1.0609037, -0.1528273, -1.8453032],
                                 [-0.0956733, 0.8300833, 0.1195768, 1.4438214],
                                 [-0.0390998, 0.3392385, 0.0488686, 0.5900609]])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestPow(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()

    def test_api_completeness_1(self):
        input_1 = torch.tensor([1.5771123, 1.0215064, 0.3217487, -1.4698992])
        output_ = torch.pow(input_1, 2)
        print(output_)
        expected = torch.tensor([2.4872832, 1.0434754, 0.1035222, 2.1606035])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_1 = torch.tensor([1., 2., 3., 4.])
        input_2 = torch.tensor([1., 2., 3., 4.])
        output_ = torch.pow(input_1, input_2)
        print(output_)
        expected = torch.tensor([1., 4., 27., 256.])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestRsqrt(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([16, 0.16])
        y = torch.rsqrt(x)
        print(y)
        expected = torch.tensor([0.2500000, 2.5000000])
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {y}"


class TestSigmoid(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-0.0786774, -1.0416820, -1.6688058, 0.3292783])
        y = torch.sigmoid(x)
        print(y)
        expected = torch.tensor([0.4803408, 0.2608256, 0.1585834, 0.5815837])
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {y}"


class TestSign(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([0.7, -1.2, 0., 2.3])
        y = torch.sign(x)
        print(y)
        expected = torch.tensor([1., -1., 0., 1.])
        assert torch.allclose(y, expected), f"expected: {expected}, but got {y}"


class TestSin(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([0.7348099, 1.6324461, -1.5990942, -0.0387336])
        y = torch.sin(x)
        print(y)
        expected = torch.tensor([0.6704461, 0.9981003, -0.9995996, -0.0387239])
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {y}"


class TestSoftmax(TestCase):

    def test_api_completeness(self):
        self.test_api_completeness_1()
        self.test_api_completeness_2()
        self.test_api_completeness_3()
        self.test_api_completeness_4()
        self.test_api_completeness_5()

    def test_api_completeness_1(self):
        input_ = torch.tensor([[-0.8172550, -0.4018371, -0.1398522],
                               [-0.9724374, 0.4756050, -0.0959570]])
        dim = None
        dtype = None
        output_ = torch.softmax(input=input_, dim=dim, dtype=dtype)
        expected = torch.tensor([[0.223027021, 0.337886781, 0.439086229],
                                 [0.130595937, 0.555656612, 0.313747495]])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_2(self):
        input_ = torch.tensor([[-0.8172550, -0.4018371, -0.1398522],
                               [-0.9724374, 0.4756050, -0.0959570]])
        dim = 1
        dtype = torch.float32
        output_ = torch.softmax(input=input_, dim=dim, dtype=dtype)
        expected = torch.tensor([[0.2230270, 0.3378868, 0.4390862],
                                 [0.1305959, 0.5556566, 0.3137475]])
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_3(self):
        input_ = torch.tensor([[-0.8172550, -0.4018371, -0.1398522],
                               [-0.9724374, 0.4756050, -0.0959570]])
        dim = 0
        dtype = torch.float64
        output_ = torch.softmax(input=input_, dim=dim, dtype=dtype)
        expected = torch.tensor([[0.5387179, 0.2937081, 0.4890280],
                                 [0.4612821, 0.7062919, 0.5109720]], dtype=torch.float64)
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_4(self):
        input_ = torch.tensor([[-0.8172550, -0.4018371, -0.1398522],
                               [-0.9724374, 0.4756050, -0.0959570]], dtype=torch.float16)
        dim = None
        dtype = torch.float16
        output_ = torch.softmax(input=input_, dim=dim, dtype=dtype)
        expected = torch.tensor([[0.22302, 0.33789, 0.43921],
                                 [0.13062, 0.55566, 0.31372]], dtype=dtype)
        assert torch.allclose(output_, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"

    def test_api_completeness_5(self):
        input_ = torch.tensor([[-0.8172550, -0.4018371, -0.1398522],
                               [-0.9724374, 0.4756050, -0.0959570]], dtype=torch.bfloat16)
        dim = None
        dtype = torch.bfloat16
        output_ = torch.softmax(input=input_, dim=dim, dtype=dtype)
        expected = torch.tensor([[0.2236328, 0.3378906, 0.4394531],
                                 [0.1308594, 0.5546875, 0.3144531]], dtype=torch.bfloat16)
        assert torch.allclose(output_.float(), expected.float(), rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {output_}"


class TestSquare(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-1.1704886, -0.5830599, 0.7463780, 0.3248016])
        y = torch.square(x)
        print(y)
        expected = torch.tensor([1.3700435, 0.3399588, 0.5570801, 0.1054961])
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {y}"


class TestTanh(TestCase):

    def test_api_completeness(self):
        x = torch.tensor([-1.1704886, -0.5830599, 0.7463780, 0.3248016])
        y = torch.tanh(x)
        print(y)
        expected = torch.tensor([-0.8244287, -0.5248859, 0.6329831, 0.3138421])
        assert torch.allclose(y, expected, rtol=1e-3, atol=1e-4), f"expected: {expected}, but got {y}"


if __name__ == "__main__":
    print(f"PYTHONPATH is:\n{os.getenv('PYTHONPATH')}")
    seed_all(1921)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, choices=['completeness', 'performance'],
                        help="test mode")

    args, _ = parser.parse_known_args()

    if args.test_mode == 'completeness':
        TestAbs().test_api_completeness()
        TestAdd().test_api_completeness()
        TestClamp().test_api_completeness()
        TestCos().test_api_completeness()
        TestDiv().test_api_completeness()
        TestExp().test_api_completeness()
        TestLog().test_api_completeness()
        TestLog2().test_api_completeness()
        TestLogicalAnd().test_api_completeness()
        TestLogicalNot().test_api_completeness()
        TestLogicalOr().test_api_completeness()
        TestMul().test_api_completeness()
        TestPow().test_api_completeness()
        TestRsqrt().test_api_completeness()
        TestSigmoid().test_api_completeness()
        TestSign().test_api_completeness()
        TestSin().test_api_completeness()
        TestSoftmax().test_api_completeness()
        TestSquare().test_api_completeness()
        TestTanh().test_api_completeness()
