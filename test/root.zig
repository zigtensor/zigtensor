const std = @import("std");

test "expect true to be true" {
    try std.testing.expect(true == true);
}
