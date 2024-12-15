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

pub const Activation = enum {
    none,
    sigmoid,
    pub fn function(self: Activation) *const fn (f32) f32 {
        return &switch (self) {
            .none => math.identity,
            .sigmoid => math.sigmoid,
        };
    }
    pub fn derivative(self: Activation) *const fn (f32) f32 {
        return &switch (self) {
            .none => math.identityp,
            .sigmoid => math.sigmoidp,
        };
    }
};

// only for 2D input dimension
pub fn Neuron() type {
    return struct {
        const T = f32;
        weights: mat.Matrix(T, 2),
        //basis_functions: mat.Matrix(*const fn (f32) f32, 2),
        activation_function: Activation = .none,
        a: mat.Matrix(T, 2),
        const Self = @This();

        pub fn get_activation(self: Self) T {
            const x = try self.a.get([2]usize{ 0, 0 });
            return self.activation_function.function()(x);
        }

        pub fn calc_activation(self: *Self, x: mat.Matrix(T, 2)) !void {
            try mat.matrix_multiplication(T, x, self.weights, &self.a);
        }

        pub fn set_f(self: *Self, activation: Activation) void {
            self.activation_function = activation;
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            for (self.weights.buf, 0..) |_, i| {
                self.weights.buf[i] = random(rand);
            }
        }

        fn random(rand: std.Random) T {
            //return @as(T, @floatFromInt(rand.intRangeAtMost(i8, -20, 20)));
            return rand.floatNorm(f32);
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

test "neuron" {
    var neuron = try Neuron().init_alloc(std.testing.allocator, 3);
    defer neuron.deinit(std.testing.allocator);

    neuron.weights.buf[0] = 1;
    neuron.weights.buf[1] = 2;
    neuron.weights.buf[2] = 4;

    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 3 });
    defer x.deinit(std.testing.allocator);

    x.buf[0] = 3;
    x.buf[1] = 2;
    x.buf[2] = 1;

    try neuron.calc_activation(x);

    try std.testing.expect(neuron.get_activation() == 3 * 1 + 2 * 2 + 1 * 4);
}

pub fn Layer() type {
    return struct {
        const T = f32;
        weights: mat.Matrix(T, 2),
        //basis_functions: mat.Matrix(*const fn (comptime type, T) T, 2),
        //activation_function: fn (T) T,
        activation_function: Activation = .none,
        forward: mat.Matrix(T, 2),
        activations: mat.Matrix(T, 2),
        neurons: []Neuron(),
        const Self = @This();

        pub fn calc_activation(self: *Self, x: mat.Matrix(T, 2)) !void {
            try mat.matrix_multiplication(T, x, self.weights, &self.activations);
        }

        pub fn set_f(self: *Self, activation: Activation) void {
            self.activation_function = activation;
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
            //return @as(T, @floatFromInt(rand.intRangeAtMost(i8, -20, 20)));
            return rand.floatNorm(f32);
        }

        pub fn reset(self: *Self) void {
            for (self.weights.buf, 0..) |_, i| {
                self.weights.buf[i] = 0;
            }
        }

        pub fn init(allocator: std.mem.Allocator, n_neurons: usize, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, n_neurons });
            const forward = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, n_neurons });
            const a = try mat.Matrix(T, 2).init(forward.buf, [2]usize{ 1, n_neurons });
            var i: usize = 0;
            var neurons = try allocator.alloc(Neuron(), n_neurons);
            while (i < n_neurons) : (i += 1) {
                const start = input_size * i;
                neurons[i] = try Neuron().init(weights.buf[start .. start + input_size], a.buf[i .. i + 1]);
            }
            return Self{ .weights = weights, .activations = a, .forward = forward, .neurons = neurons };
        }

        pub fn init_forward(allocator: std.mem.Allocator, n_neurons: usize, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, n_neurons });
            const forward = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, n_neurons + 1 });
            const a = try mat.Matrix(T, 2).init(forward.buf[1..], [2]usize{ 1, n_neurons });
            forward.buf[0] = 1;
            var i: usize = 0;
            var neurons = try allocator.alloc(Neuron(), n_neurons);
            while (i < n_neurons) : (i += 1) {
                const start = input_size * i;
                neurons[i] = try Neuron().init(weights.buf[start .. start + input_size], a.buf[i .. i + 1]);
            }
            return Self{ .weights = weights, .activations = a, .forward = forward, .neurons = neurons };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.weights.deinit(allocator);
            self.forward.deinit(allocator);
            allocator.free(self.neurons);
        }
    };
}

test "layer" {
    var layer = try Layer().init(std.testing.allocator, 2, 3);
    defer layer.deinit(std.testing.allocator);

    layer.weights.buf[0] = 1;
    layer.weights.buf[1] = 2;
    layer.weights.buf[2] = 4;
    layer.weights.buf[3] = 1;
    layer.weights.buf[4] = 3;
    layer.weights.buf[5] = 9;

    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 3 });
    defer x.deinit(std.testing.allocator);

    x.buf[0] = 3;
    x.buf[1] = 2;
    x.buf[2] = 1;

    try layer.calc_activation(x);

    try std.testing.expect(layer.activations.buf[0] == 3 * 1 + 2 * 2 + 1 * 4);
    try std.testing.expect(layer.activations.buf[1] == 3 * 1 + 2 * 3 + 1 * 9);
}

pub fn Network() type {
    return struct {
        layers: []Layer(),
        gradient: []Layer(),
        count: f32 = 0,
        diff_acc: f32 = 0,
        y: mat.Matrix(f32, 2),
        learning_rate: f32 = 1,
        eps: f32 = 1e-7,
        const Self = @This();

        pub fn feed_forward(self: *Self, x: mat.Matrix(f32, 2), expected: f32) !mat.Matrix(f32, 2) {
            var in = x;
            for (self.layers, 0..) |_, i| {
                try mat.matrix_multiplication(f32, in, self.layers[i].weights, @constCast(&self.layers[i].activations));
                in = try mat.Matrix(f32, 2).init(self.layers[i].forward.buf, [2]usize{ 1, self.layers[i].forward.buf.len });
            }
            @memcpy(self.y.buf, in.buf);
            const cost = std.math.pow(f32, self.y.buf[0] - expected, 2);
            for (self.layers, 0..) |layer, i| {
                for (layer.weights.buf, 0..) |w, j| {
                    const old = w;
                    self.layers[i].weights.buf[j] = w + self.eps;
                    in = x;
                    for (self.layers, 0..) |_, k| {
                        try mat.matrix_multiplication(f32, in, self.layers[k].weights, @constCast(&self.layers[k].activations));
                        in = try mat.Matrix(f32, 2).init(self.layers[k].forward.buf, [2]usize{ 1, self.layers[k].forward.buf.len });
                    }
                    in = self.layers[self.layers.len - 1].activations;
                    self.layers[i].weights.buf[j] = old;

                    const cost_ = std.math.pow(f32, in.buf[0] - expected, 2);
                    self.gradient[i].weights.buf[j] += cost_;
                }
            }
            self.diff_acc += cost;
            self.count += 1;
            return self.y;
        }

        pub fn use_training(self: *Self) void {
            const cost = self.diff_acc / 2;
            std.log.debug("cost:\t{d}", .{cost});
            for (self.gradient, 0..) |layer, i| {
                for (layer.weights.buf, 0..) |w, j| {
                    const lim = ((w / 2) - cost) / self.eps;
                    self.layers[i].weights.buf[j] -= self.learning_rate * lim;
                }

                self.gradient[i].reset();
            }
            self.diff_acc = 0;
            self.count = 0;
        }

        pub fn err(self: *Self) f32 {
            return self.diff_acc / 2;
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            for (self.layers) |layer| {
                @constCast(&layer).randomize(rand);
            }
        }

        pub fn reset_gradient(self: *Self) void {
            for (self.gradient, 0..) |_, i| {
                self.layers[i].reset();
            }
        }

        pub fn init(allocator: std.mem.Allocator, input_size: usize, layer_nodes_amount: []usize) !Self {
            if (layer_nodes_amount.len < 1) {
                @panic("");
            }

            var layers = try allocator.alloc(Layer(), layer_nodes_amount.len);
            layers[0] = if (1 >= layer_nodes_amount.len)
                try Layer().init(allocator, layer_nodes_amount[0], input_size)
            else
                try Layer().init_forward(allocator, layer_nodes_amount[0], input_size);
            layers[0].set_f(Activation.sigmoid);
            var i: usize = 1;
            while (i < layer_nodes_amount.len) : (i += 1) {
                layers[i] = if (i + 1 >= layer_nodes_amount.len)
                    try Layer().init(allocator, layer_nodes_amount[i], layer_nodes_amount[i - 1] + 1)
                else
                    try Layer().init_forward(allocator, layer_nodes_amount[i], layer_nodes_amount[i - 1] + 1);

                if (i + 1 < layer_nodes_amount.len) {
                    layers[i].set_f(Activation.sigmoid);
                }
            }

            var gradients = try allocator.alloc(Layer(), layer_nodes_amount.len);
            gradients[0] = if (1 >= layer_nodes_amount.len)
                try Layer().init(allocator, layer_nodes_amount[0], input_size)
            else
                try Layer().init_forward(allocator, layer_nodes_amount[0], input_size);
            gradients[0].reset();
            i = 1;
            while (i < layer_nodes_amount.len) : (i += 1) {
                gradients[i] = if (i + 1 >= layer_nodes_amount.len)
                    try Layer().init(allocator, layer_nodes_amount[i], layer_nodes_amount[i - 1] + 1)
                else
                    try Layer().init_forward(allocator, layer_nodes_amount[i], layer_nodes_amount[i - 1] + 1);
                gradients[i].reset();
            }

            const y = try mat.Matrix(f32, 2).init_alloc(allocator, [2]usize{ layer_nodes_amount[layer_nodes_amount.len - 1], 1 });

            return Self{ .layers = layers, .gradient = gradients, .y = y };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            for (self.layers) |layer| {
                layer.deinit(allocator);
            }
            for (self.gradient) |layer| {
                layer.deinit(allocator);
            }
            allocator.free(self.layers);
            allocator.free(self.gradient);
            self.y.deinit(allocator);
        }
    };
}

test "neural network" {
    var nn = try Network().init(std.testing.allocator, 1, @constCast(&[_]usize{1}));
    defer nn.deinit(std.testing.allocator);
    nn.learning_rate = 1e-2;

    // 2x = y
    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 1 });
    defer x.deinit(std.testing.allocator);
    x.buf[0] = 5;

    nn.layers[0].weights.buf[0] = 0;

    var i: usize = 0;
    _ = try nn.feed_forward(x, 10);
    var cost = nn.diff_acc / nn.count;
    nn.use_training();
    while (i < 1e2) : (i += 1) {
        std.log.warn("c: {d}\tw: {d}", .{ cost, nn.layers[0].weights.buf[0] });
        _ = try nn.feed_forward(x, 10);
        const new_cost = nn.diff_acc / nn.count;
        //try std.testing.expect(cost >= new_cost);
        cost = new_cost;
        nn.use_training();
    }
}
