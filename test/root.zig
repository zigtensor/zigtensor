const std = @import("std");

const type_tests = @import("./core/types_test.zig");
const tensor_tests = @import("./tensor/tensor_test.zig");
const ops_tests = @import("./tensor/ops_test.zig");

test "expect true to be true" {
    try std.testing.expect(true == true);
    std.testing.refAllDecls(type_tests);
    std.testing.refAllDecls(tensor_tests);
    std.testing.refAllDecls(ops_tests);
}
