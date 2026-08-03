import msadapter


# Used for testing and logging
dtype_abbrs = {
    msadapter.bfloat16: "bf16",
    msadapter.float64: "f64",
    msadapter.float32: "f32",
    msadapter.float16: "f16",
    msadapter.complex32: "c32",
    msadapter.complex64: "c64",
    msadapter.complex128: "c128",
    msadapter.int8: "i8",
    msadapter.int16: "i16",
    msadapter.int32: "i32",
    msadapter.int64: "i64",
    msadapter.bool: "b8",
    msadapter.uint8: "u8",
    msadapter.uint16: "u16",
    msadapter.uint32: "u32",
    msadapter.uint64: "u64",
}
