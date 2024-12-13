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

        pub fn init_alloc(allocator: std.mem.Allocator, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, 1 });
            const a = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, 1 });
            return Self{ .weights = weights, .a = a };
        }

        pub fn init(weights: []T, a: []T) !Self {
            return Self{ .weights = try mat.Matrix(T, 2).init(weights, [2]usize{ weights.len, 1 }), .a = try mat.Matrix(T, 2).init(a, [2]usize{ 1, 1 }) };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.weights.deinit(allocator);
            self.a.deinit(allocator);
        }
    };
}

pub fn Layer() type {
    return struct {
        const T = f32;
        weights: mat.Matrix(T, 2),
        //basis_functions: mat.Matrix(*const fn (comptime type, T) T, 2),
        //activation_function: fn (T) T,
        activation_function: bool = false,
        activations: mat.Matrix(T, 2),
        neurons: []Neuron(),
        const Self = @This();

        pub fn calc_activation(self: *Self, x: mat.Matrix(T, 2)) !void {
            try mat.matrix_multiplication(T, x, self.weights, &self.activations);
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

        pub fn init(allocator: std.mem.Allocator, n_neurons: usize, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, n_neurons });
            const a = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, n_neurons });
            var i: usize = 0;
            var neurons = try allocator.alloc(Neuron(), n_neurons);
            while (i < n_neurons) : (i += 1) {
                const start = input_size * i;
                neurons[i] = try Neuron().init(weights.buf[start .. start + input_size], a.buf[i .. i + 1]);
            }
            return Self{ .weights = weights, .activations = a, .neurons = neurons };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.weights.deinit(allocator);
            self.activations.deinit(allocator);
            allocator.free(self.neurons);
        }
    };
}

pub fn Network(n_layers: comptime_int) type {
    return struct {
        layers: [n_layers]Layer(),
        const Self = @This();

        pub fn feed_forward(self: Self, x: mat.Matrix(f32, 2)) !mat.Matrix(f32, 2) {
            var in = x;
            var i: usize = 0;
            while (i < self.layers.len) : (i += 1) {
                try mat.matrix_multiplication(f32, in, self.layers[i].weights, @constCast(&self.layers[i].activations));
                in = try mat.Matrix(f32, 2).init(self.layers[i].activations.buf, [2]usize{ 1, self.layers[i].activations.buf.len });
            }
            return in;
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            for (self.layers) |layer| {
                layer.randomize(rand);
            }
        }

        pub fn init(allocator: std.mem.Allocator, input_size: usize, layer_nodes_amount: [n_layers]usize) !Self {
            if (layer_nodes_amount.len < 1) {
                @panic("");
            }

            var layers: [n_layers]Layer() = undefined;
            layers[0] = try Layer().init(allocator, layer_nodes_amount[0], input_size);
            layers[0].set_f(false);
            var i: usize = 1;
            while (i < layer_nodes_amount.len) : (i += 1) {
                layers[i] = try Layer().init(allocator, layer_nodes_amount[i], layer_nodes_amount[i - 1]);
                layers[i].set_f(false);
            }
            return Self{ .layers = layers };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            for (self.layers) |layer| {
                layer.deinit(allocator);
            }
        }
    };
}
