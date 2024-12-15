// LANGuage IDentification - To identify languages.
// Copyright (C) 2024  João Augusto Costa Branco Marado Torres
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
// for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>
const std = @import("std");

pub fn Matrix(comptime T: type, dimensions: comptime_int) type {
    return struct {
        buf: []T,
        sz: Index(dimensions) = undefined,
        const Self = @This();

        fn index(self: Self, position: Index(dimensions)) usize {
            var i: usize = position[dimensions - 1];
            var d: usize = dimensions - 2;
            while (d >= 0) : (d -= 1) {
                i *= self.sz[d];
                i += position[d];

                if (d == 0) break; // prevent integer overflow
            }
            return i;
        }

        pub fn get(self: Self, position: Index(dimensions)) !T {
            return self.buf[self.index(position)];
        }

        pub fn set(self: *Self, position: Index(dimensions), val: T) !void {
            self.buf[self.index(position)] = val;
        }

        pub fn fill(self: *Self, v: T) void {
            var i: usize = blk: {
                var mul: usize = 1;
                for (self.sz) |value| mul *= value;
                break :blk mul;
            };

            while (i >= 0) : (i -= 1) self.buf[i] = v;
        }

        pub fn get_dimensions() comptime_int {
            return dimensions;
        }

        pub fn init_alloc(allocator: std.mem.Allocator, size: Index(dimensions)) !Self {
            const buf = try allocator.alloc(T, blk: {
                var mul: usize = 1;
                for (size) |value| mul *= value;
                break :blk mul;
            });
            return Self{ .buf = buf, .sz = size };
        }

        pub fn init(buf: []T, size: Index(dimensions)) !Self {
            return Self{ .buf = buf, .sz = size };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.buf);
        }
    };
}

pub fn Index(dimensions: comptime_int) type {
    return [dimensions]usize;
}

// to work with 2 dimensions only right now
//pub fn matrix_multiplication(comptime T: type, comptime dimensions: comptime_int, x: Matrix(T, dimensions), y: Matrix(T, dimensions), z: Matrix(T, dimensions)) !void {
pub fn matrix_multiplication(comptime T: type, x: Matrix(T, 2), y: Matrix(T, 2), res: *Matrix(T, 2)) !void {
    const sizes_x = x.sz;
    const sizes_y = y.sz;
    const sizes_res = res.sz;

    if (sizes_x[1] != sizes_y[0]) @panic("");
    if (sizes_x[0] != sizes_res[0]) @panic("");
    if (sizes_y[1] != sizes_res[1]) @panic("");

    var i: usize = 0;
    while (i < sizes_res[0]) : (i += 1) {
        var j: usize = 0;
        while (j < sizes_res[1]) : (j += 1) {
            try res.set([2]usize{ i, j }, blk: {
                var sum: T = 0;
                var k: usize = 0;
                while (k < sizes_x[1]) : (k += 1) sum += try x.get([2]usize{ i, k }) * try y.get([2]usize{ k, j });
                break :blk sum;
            });
        }
    }
}

test "Matrix multiplication" {
    const allocator = std.testing.allocator;

    const valuesA = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const sizeA: Index(2) = [2]usize{ 3, 2 };
    const matA = try Matrix(f32, 2).init(@constCast(&valuesA), sizeA);

    const valuesB = [_]f32{
        7.0, 8.0,
        9.0, 10.0,
    };
    const sizeB: Index(2) = [2]usize{ 2, 2 };
    const matB = try Matrix(f32, 2).init(@constCast(&valuesB), sizeB);

    const sizeRes: Index(2) = [2]usize{ 3, 2 };
    var result = try Matrix(f32, 2).init_alloc(allocator, sizeRes);
    defer result.deinit(allocator);

    try matrix_multiplication(f32, matA, matB, &result);

    const expected = [_]f32{
        1 * 7 + 4 * 8,  2 * 7 + 5 * 8,  3 * 7 + 6 * 8,
        1 * 9 + 4 * 10, 2 * 9 + 5 * 10, 3 * 9 + 6 * 10,
    };

    for (expected, 0..) |val, i| {
        try std.testing.expect(std.math.approxEqAbs(f32, val, result.buf[i], 1e-4));
    }
}
