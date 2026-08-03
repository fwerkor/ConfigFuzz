import re
import os
import sys
import inspect
import unittest
import expecttest
from numbers import Number
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import torch
from torch import inf

from torch_npu.testing.common_utils import is_iterable

# Environment variables set in ci script.
IS_IN_CI = os.getenv('IN_CI') == '1'
TEST_REPORT_PATH = os.getenv("TEST_REPORT_PATH", "test-reports")


def run_tests():
    argv = sys.argv
    if IS_IN_CI:
        # import here so that non-CI doesn't need xmlrunner installed
        import xmlrunner
        filename = inspect.getfile(sys._getframe(1))
        strip_py = re.sub(r'.py$', '', filename)
        test_filename = re.sub('/', r'.', strip_py)
        test_report_path = os.path.join(TEST_REPORT_PATH, test_filename)
        verbose = '--verbose' in argv or '-v' in argv
        if verbose:
            print(f'Test results will be stored in {test_report_path}')
        unittest.main(argv=argv, testRunner=xmlrunner.XMLTestRunner(output=test_report_path,
                                                                    verbosity=2 if verbose else 1))
    else:
        unittest.main(argv=argv)


class TestCase(expecttest.TestCase):
    _precision = 1e-5
    maxDiff = None
    exact_dtype = False

    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, prec):
        self._precision = prec

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if exact_dtype is None:
            exact_dtype = self.exact_dtype

        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

    def _assertNumberEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self._assertNumberEqual(x.item(), y, prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)

        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self._assertNumberEqual(x, y.item(), prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)

        else:
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)

    def _assertBoolEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if isinstance(x, torch.Tensor) and isinstance(y, np.bool_):
            self._assertBoolEqual(x.item(), y, prec=prec, message=message,
                                  allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif isinstance(y, torch.Tensor) and isinstance(x, np.bool_):
            self._assertBoolEqual(x, y.item(), prec=prec, message=message,
                                  allow_inf=allow_inf, exact_dtype=exact_dtype)
        else:
            super(TestCase, self).assertEqual(x, y, message)

    def _assertTensorsEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
        super(TestCase, self).assertEqual(x.is_quantized, y.is_quantized, message)
        if x.is_sparse:
            x = self.safeCoalesce(x)
            y = self.safeCoalesce(y)
            self._assert_tensor_equal(x._indices(), y._indices(), message, exact_dtype, allow_inf, prec)
            self._assert_tensor_equal(x._values(), y._values(), message, exact_dtype, allow_inf, prec)
        elif x.is_quantized and y.is_quantized:
            self.assertEqual(x.qscheme(), y.qscheme(), prec=prec,
                                message=message, allow_inf=allow_inf,
                                exact_dtype=exact_dtype)
            if x.qscheme() == torch.per_tensor_affine:
                self.assertEqual(x.q_scale(), y.q_scale(), prec=prec,
                                    message=message, allow_inf=allow_inf,
                                    exact_dtype=exact_dtype)
                self.assertEqual(x.q_zero_point(), y.q_zero_point(),
                                    prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif x.qscheme() == torch.per_channel_affine:
                self.assertEqual(x.q_per_channel_scales(), y.q_per_channel_scales(), prec=prec,
                                    message=message, allow_inf=allow_inf,
                                    exact_dtype=exact_dtype)
                self.assertEqual(x.q_per_channel_zero_points(), y.q_per_channel_zero_points(),
                                    prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)
                self.assertEqual(x.q_per_channel_axis(), y.q_per_channel_axis(),
                                    prec=prec, message=message)
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.int_repr().to(torch.int32),
                                y.int_repr().to(torch.int32), prec=prec,
                                message=message, allow_inf=allow_inf,
                                exact_dtype=exact_dtype)
        else:
            self._assert_tensor_equal(x, y, message, exact_dtype, allow_inf, prec)

        def _assertEqual(x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
            if isinstance(x, Number) or isinstance(y, Number):
                self._assertNumberEqual(x, y, prec=prec, message=message,
                                        allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif isinstance(x, np.bool_) or isinstance(y, np.bool_):
                self._assertBoolEqual(x, y, prec=prec, message=message,
                                    allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                self._assertTensorsEqual(x, y, prec=prec, message=message,
                                        allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif isinstance(x, (str, bytes)) and isinstance(y, (str, bytes)):
                super(TestCase, self).assertEqual(x, y, message)
            elif type(x) == set and type(y) == set:
                super(TestCase, self).assertEqual(x, y, message)
            elif isinstance(x, dict) and isinstance(y, dict):
                if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                    _assertEqual(x.items(), y.items(), prec=prec,
                                 message=message, allow_inf=allow_inf,
                                 exact_dtype=exact_dtype)
                else:
                    _assertEqual(set(x.keys()), set(y.keys()), prec=prec,
                                 message=message, allow_inf=allow_inf,
                                 exact_dtype=exact_dtype)
                    key_list = list(x.keys())
                    _assertEqual([x[k] for k in key_list],
                                 [y[k] for k in key_list],
                                 prec=prec, message=message,
                                 allow_inf=allow_inf, exact_dtype=exact_dtype)
            elif is_iterable(x) and is_iterable(y):
                super(TestCase, self).assertEqual(len(x), len(y), message)
                for x_, y_ in zip(x, y):
                    _assertEqual(x_, y_, prec=prec, message=message,
                                 allow_inf=allow_inf, exact_dtype=exact_dtype)
            else:
                super(TestCase, self).assertEqual(x, y, message)

        _assertEqual(x, y, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)


    def assertRtolEqual(self, x, y, prec=1.e-4, prec16=1.e-3, auto_trans_dtype=False, message=None):

        def _assertRtolEqual(x, y, prec, prec16, message):
            def compare_res(pre, minimum):
                diff = y - x
                # check that NaNs are in the same locations
                nan_mask = np.isnan(x)
                if not np.equal(nan_mask, np.isnan(y)).all():
                    self.fail(message)
                if nan_mask.any():
                    diff[nan_mask] = 0
                result = np.abs(diff)
                deno = np.maximum(np.abs(x), np.abs(y))
                result_atol = np.less_equal(result, pre)
                result_rtol = np.less_equal(result / np.add(deno, minimum), pre)
                if not result_rtol.all() and not result_atol.all():
                    if np.sum(result_rtol == False) > size * pre and np.sum(result_atol == False) > size * pre:
                        self.fail("result error")

            minimum16 = 6e-8
            minimum = 10e-10

            if isinstance(x, Sequence) and isinstance(y, Sequence):
                for x_, y_ in zip(x, y):
                    _assertRtolEqual(x_, y_, prec, prec16, message)
                return

            if isinstance(x, torch.Tensor) and isinstance(y, Sequence):
                y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
            elif isinstance(x, Sequence) and isinstance(y, torch.Tensor):
                x = torch.as_tensor(x, dtype=y.dtype, device=y.device)

            if torch.is_tensor(x) and torch.is_tensor(y):
                if auto_trans_dtype:
                    x = x.to(y.dtype)
                x = x.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
            elif isinstance(x, Number) and isinstance(y, Number):
                x = np.array(x)
                y = np.array(y)

            size = x.size
            if (x.shape != y.shape):
                self.fail("shape error")
            if (x.dtype != y.dtype):
                self.fail("dtype error")
            dtype_list = [np.bool_, np.uint16, np.int16, np.int32, np.float16,
                        np.float32, np.int8, np.uint8, np.int64, np.float64]
            if x.dtype not in dtype_list:
                self.fail("required dtype in [np.bool_, np.uint16, np.int16, " +
                        "np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]")
            if x.dtype == np.bool_:
                result = np.equal(x, y)
                if not result.all():
                    self.fail("result error")
            elif (x.dtype == np.float16):
                compare_res(prec16, minimum16)
            elif (x.dtype in [np.float32, np.int8, np.uint8, np.uint16, np.int16, np.int32, np.int64, np.float64]):
                compare_res(prec, minimum)
            else:
                self.fail("required numpy object")

        _assertRtolEqual(x, y, prec, prec16, message)
