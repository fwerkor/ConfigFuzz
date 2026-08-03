from enum import Enum


__all__ = ['strided', 'sparse_coo', 'sparse_csc', 'sparse_bsc', 'sparse_bsr', 'sparse_csr', 'nested_tensor', 'undefined', '_mkldnn']

# layout
class Layout(Enum):
    STRIDED = "strided"
    SPARSE_COO = "sparse_coo"
    SPARSE_CSR = "sparse_csr"
    MKLDNN = "_mkldnn"
    SPARSE_CSC = "sparse_csc"
    SPARSE_BSR = "sparse_bsr"
    SPARSE_BSC = "sparse_bsc"
    NESTED_TENSOR = "_nested_tensor"
    UNDEFINED = "undefined"

    def __str__(self):
        return f"msadapter.{self.value}"

strided = Layout.STRIDED
sparse_coo = Layout.SPARSE_COO
sparse_csr = Layout.SPARSE_CSR
sparse_csc = Layout.SPARSE_CSC
sparse_bsr = Layout.SPARSE_BSR
sparse_bsc = Layout.SPARSE_BSC
nested_tensor = Layout.NESTED_TENSOR
undefined = Layout.UNDEFINED
_mkldnn = Layout.MKLDNN