const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zigtensor_module = b.addModule("zigtensor", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{ .name = "zigtensor", .linkage = .static, .root_module = b.createModule(.{
        .root_source_file = b.path("./src/lib.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.installArtifact(lib);

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("test/root.zig"),
        .target = target,
        .optimize = optimize,
    }) });

    unit_tests.root_module.addImport("zigtensor", zigtensor_module);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    b.default_step.dependOn(test_step);
}
