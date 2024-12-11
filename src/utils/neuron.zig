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
const mat = @import("./matrix.zig");
const math = @import("./math.zig");

// only for 2D input dimension
pub fn Neuron() type {
    return struct {
        const T = f32;
        weights: mat.Matrix(T, 2),
        //basis_functions: mat.Matrix(*const fn (comptime type, T) T, 2),
        //activation_function: fn (T) T,
        activation_function: bool = false,
        a: mat.Matrix(T, 2),
        const Self = @This();

        pub fn get_activation(self: Self) T {
            const x = try self.a.get([2]usize{ 0, 0 });
            return if (self.activation_function) math.sigmoid(T, x) else x;
        }

        pub fn calc_activation(self: *Self, x: mat.Matrix(T, 2)) !void {
            try mat.matrix_multiplication(T, x, self.weights, &self.a);
        }

        pub fn set_f(self: *Self, f: bool) void {
            self.activation_function = f;
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            const sz = self.weights.sz;
            var i: usize = 0;
            while (i < sz[0]) : (i += 1) {
                var j: usize = 0;
                while (j < sz[1]) : (j += 1) {
                    const pos = [2]usize{ i, j };
                    const r = random(rand);
                    try self.weights.set(pos, r);
                }
            }
        }

        fn random(rand: std.Random) T {
            return @as(T, @floatFromInt(rand.intRangeAtMost(i8, -20, 20)));
        }

        pub fn init(allocator: std.mem.Allocator, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init(allocator, [2]usize{ input_size, 1 });
            const a = try mat.Matrix(T, 2).init(allocator, [2]usize{ 1, 1 });
            return Self{ .weights = weights, .a = a };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.weights.deinit(allocator);
            self.a.deinit(allocator);
        }
    };
}
