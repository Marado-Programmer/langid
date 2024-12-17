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
    reLU,
    rosenblatt,
    tanh,
    pub fn function(self: Activation, x: f32) f32 {
        return switch (self) {
            .none => math.identity(x),
            .sigmoid => math.sigmoid(x),
            .reLU => math.reLU(x),
            .rosenblatt => math.rosenblatt(x),
            .tanh => math.tanh(x),
        };
    }
    pub fn derivative(self: Activation, x: f32) f32 {
        return switch (self) {
            .none => math.identityp(x),
            .sigmoid => math.sigmoidp(x),
            .reLU => math.reLUp(x),
            .rosenblatt => math.rosenblattp(x),
            .tanh => math.tanhp(x),
        };
    }
};

// only for 2D input dimension
pub fn Neuron() type {
    return struct {
        const T = f32;
        weights: mat.Matrix(T, 2),
        // basis_functions: mat.Matrix(Activation, 2),
        activation_function: Activation = .none,
        y: mat.Matrix(T, 2),
        const Self = @This();

        pub fn get_activation(self: Self) T {
            return self.activation_function.function(self.get_y());
        }

        pub fn calculate_activation(self: *Self, x: mat.Matrix(T, 2)) T {
            try mat.matrix_multiplication(T, x, self.weights, &self.y);
            return self.get_activation();
        }

        pub fn get_y(self: Self) T {
            return self.y.buf[0];
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            for (self.weights.buf, 0..) |_, i| {
                self.weights.buf[i] = random(rand);
            }
        }

        fn random(rand: std.Random) T {
            return rand.floatNorm(T);
        }

        pub fn fill(self: *Self, x: T) void {
            for (self.weights.buf, 0..) |_, i| {
                self.weights.buf[i] = x;
            }
        }

        pub fn reset(self: *Self) void {
            self.fill(0);
        }

        pub fn init_alloc(allocator: std.mem.Allocator, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, 1 });
            const y = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, 1 });
            return Self{ .weights = weights, .y = y };
        }

        pub fn init(weights: []T, y: []T) !Self {
            const weights_matrix = try mat.Matrix(T, 2).init(weights, [2]usize{ weights.len, 1 });
            const y_matrix = try mat.Matrix(T, 2).init(y, [2]usize{ 1, 1 });
            return Self{ .weights = weights_matrix, .y = y_matrix };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            self.weights.deinit(allocator);
            self.y.deinit(allocator);
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

    try std.testing.expect(neuron.calculate_activation(x) == 3 * 1 + 2 * 2 + 1 * 4);
}

pub fn Layer() type {
    return struct {
        const T = f32;
        weights: mat.Matrix(T, 2),
        forward: mat.Matrix(T, 2),
        activations: mat.Matrix(T, 2),
        neurons: []Neuron(),
        const Self = @This();

        pub fn calculate_activations(self: *Self, x: mat.Matrix(T, 2)) !*const mat.Matrix(T, 2) {
            try mat.matrix_multiplication(T, x, self.weights, &self.activations);
            return &self.activations;
        }

        pub fn set_activation_functions(self: *Self, f: Activation) void {
            for (self.neurons, 0..) |_, i| {
                self.neurons[i].activation_function = f;
            }
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            for (self.neurons, 0..) |_, i| {
                self.neurons[i].randomize(rand);
            }
        }

        pub fn fill(self: *Self, x: T) void {
            for (self.neurons, 0..) |_, i| {
                self.neurons[i].fill(x);
            }
        }

        pub fn reset(self: *Self) void {
            self.fill(0);
        }

        pub fn init(allocator: std.mem.Allocator, n_neurons: usize, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, n_neurons });
            const forward = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, n_neurons });
            const a = try mat.Matrix(T, 2).init(forward.buf, [2]usize{ 1, n_neurons });
            var neurons = try allocator.alloc(Neuron(), n_neurons);
            var i: usize = 0;
            while (i < n_neurons) : (i += 1) {
                const start = input_size * i;
                const end = start + input_size;
                neurons[i] = try Neuron().init(weights.buf[start..end], a.buf[i .. i + 1]);
            }
            return Self{ .weights = weights, .activations = a, .forward = forward, .neurons = neurons };
        }

        pub fn init_with_bias(allocator: std.mem.Allocator, n_neurons: usize, input_size: usize) !Self {
            const weights = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ input_size, n_neurons });
            const forward = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ 1, n_neurons + 1 });
            const a = try mat.Matrix(T, 2).init(forward.buf[1..], [2]usize{ 1, n_neurons });
            forward.buf[0] = 1;
            var neurons = try allocator.alloc(Neuron(), n_neurons);
            var i: usize = 0;
            while (i < n_neurons) : (i += 1) {
                const start = input_size * i;
                const end = start + input_size;
                neurons[i] = try Neuron().init(weights.buf[start..end], a.buf[i .. i + 1]);
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

    const a = try layer.calculate_activations(x);

    try std.testing.expect(a.buf[0] == 3 * 1 + 2 * 2 + 1 * 4);
    try std.testing.expect(a.buf[1] == 3 * 1 + 2 * 3 + 1 * 9);
}

pub const LearningMethod = enum {
    basic,
    backpropagate,
};

pub fn Network() type {
    return struct {
        const T = f32;
        layers: []Layer(),
        gradient: []Layer(),
        y: mat.Matrix(T, 2),
        learning_rate: T = 1,
        learning_rate_decay: T = 0.99,
        learning_method: LearningMethod = .backpropagate,
        batch: bool = true,
        eps: T = 1e-7,
        count: usize = 0,
        error_accumulator: T = 0,
        const Self = @This();

        fn learn(self: *Self, x: mat.Matrix(T, 2)) !void {
            var in = x;
            for (self.layers, 0..) |_, i| {
                _ = try self.layers[i].calculate_activations(in);
                in = self.layers[i].forward;
            }
            @memcpy(self.y.buf, in.buf);
        }

        pub fn feed_forward(self: *Self, x: mat.Matrix(T, 2), expected: T) !*const mat.Matrix(T, 2) {
            try self.learn(x);

            const e = std.math.pow(T, self.y.buf[0] - expected, 2);
            self.error_accumulator += e;
            self.count += 1;

            switch (self.learning_method) {
                .basic => {
                    for (self.layers, 0..) |layer, i| {
                        for (layer.weights.buf, 0..) |w, j| {
                            self.layers[i].weights.buf[j] = w + self.eps;
                            try self.learn(x);
                            self.layers[i].weights.buf[j] = w;

                            const y = self.layers[self.layers.len - 1].activations;
                            self.gradient[i].weights.buf[j] += std.math.pow(T, y.buf[0] - expected, 2);
                        }
                    }
                },
                .backpropagate => {
                    var last = self.gradient.len - 1;
                    for (self.layers[last].neurons, 0..) |neuron, i| {
                        const delta = (neuron.y.buf[0] - expected);
                        self.gradient[last].activations.buf[i] = delta;

                        const previous_layer = if (last <= 0) x else self.layers[last - 1].forward;
                        for (neuron.weights.buf, 0..) |_, j| {
                            try self.gradient[last].weights.set([2]usize{ j, 0 }, previous_layer.buf[j] * delta);
                        }
                    }
                    if (last > 0) {
                        last -= 1;
                        while (last >= 0) : (last -= 1) {
                            for (self.layers[last].neurons, 0..) |neuron, i| {
                                const a = neuron.activation_function.derivative(neuron.y.buf[0]);
                                const sum = blk: {
                                    var add: T = 0;
                                    for (self.layers[last + 1].neurons, 0..) |prev_neuron, j| {
                                        const delta = self.gradient[last + 1].activations.buf[j];
                                        add += prev_neuron.weights.buf[i] * delta;
                                    }
                                    break :blk add;
                                };
                                const delta = a * sum;
                                self.gradient[last].activations.buf[i] = delta;

                                const previous_layer = if (last <= 0) x else self.layers[last - 1].forward;
                                for (neuron.weights.buf, 0..) |_, j| {
                                    try self.gradient[last].weights.set([2]usize{ j, 0 }, previous_layer.buf[j] * delta);
                                }
                            }

                            if (last <= 0) {
                                break;
                            }
                        }
                    }
                },
            }

            if (!self.batch) {
                self.use_training();
            }

            return &self.y;
        }

        pub fn use_training(self: *Self) void {
            const count: f32 = @as(f32, @floatFromInt(self.count));
            for (self.gradient, 0..) |layer, i| {
                for (layer.weights.buf, 0..) |w, j| {
                    self.layers[i].weights.buf[j] += switch (self.learning_method) {
                        .basic => blk: {
                            const lim = ((w / count) - self.err()) / self.eps;
                            break :blk -self.learning_rate * lim;
                        },
                        .backpropagate => -self.learning_rate * (w / count),
                    };
                }
                self.gradient[i].reset();
            }
            self.learning_rate *= self.learning_rate_decay;
            self.error_accumulator = 0;
            self.count = 0;
        }

        pub fn err(self: *Self) T {
            const count: f32 = @as(f32, @floatFromInt(self.count));
            return self.error_accumulator / count;
        }

        pub fn randomize(self: *Self, rand: std.Random) void {
            for (self.layers, 0..) |_, i| {
                self.layers[i].randomize(rand);
            }
        }

        pub fn reset_gradient(self: *Self) void {
            for (self.gradient, 0..) |_, i| {
                self.gradient[i].reset();
            }
        }

        pub fn init(allocator: std.mem.Allocator, input_size: usize, layer_nodes_amount: []usize) !Self {
            if (layer_nodes_amount.len < 1) {
                @panic("");
            }

            var layers = try allocator.alloc(Layer(), layer_nodes_amount.len);
            var gradient = try allocator.alloc(Layer(), layer_nodes_amount.len);
            var i: usize = 0;
            while (i < layer_nodes_amount.len) : (i += 1) {
                const in_size = if (i == 0) input_size else layer_nodes_amount[i - 1] + 1;
                layers[i] = if (i + 1 >= layer_nodes_amount.len)
                    try Layer().init(allocator, layer_nodes_amount[i], in_size)
                else
                    try Layer().init_with_bias(allocator, layer_nodes_amount[i], in_size);
                gradient[i] = if (i + 1 >= layer_nodes_amount.len)
                    try Layer().init(allocator, layer_nodes_amount[i], in_size)
                else
                    try Layer().init_with_bias(allocator, layer_nodes_amount[i], in_size);
            }

            const y = try mat.Matrix(T, 2).init_alloc(allocator, [2]usize{ layer_nodes_amount[layer_nodes_amount.len - 1], 1 });

            return Self{ .layers = layers, .gradient = gradient, .y = y };
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
    var y = try nn.feed_forward(x, 10);
    nn.use_training();
    while (i < 1e2) : (i += 1) {
        std.log.warn("y: {d}\tw: {d}", .{ y.buf[0], nn.layers[0].weights.buf[0] });
        y = try nn.feed_forward(x, 10);
        //const new_cost = nn.diff_acc / nn.count;
        //try std.testing.expect(cost >= new_cost);
        //cost = new_cost;
        nn.use_training();
    }
}

test "neural network 2 1 neuron layers" {
    var nn = try Network().init(std.testing.allocator, 1, @constCast(&[_]usize{ 1, 1 }));
    defer nn.deinit(std.testing.allocator);
    nn.learning_rate = 1e-2;

    // 2x = y
    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 1 });
    defer x.deinit(std.testing.allocator);
    x.buf[0] = 5;

    nn.layers[0].weights.buf[0] = 0;
    nn.layers[1].weights.buf[0] = 0;

    var i: usize = 0;
    var y = try nn.feed_forward(x, 10);
    nn.use_training();
    while (i < 1e2) : (i += 1) {
        std.log.warn("y: {d}\tw1: {d} w2: {}", .{ y.buf[0], nn.layers[0].weights.buf[0], nn.layers[1].weights.buf[0] });
        y = try nn.feed_forward(x, 10);
        nn.use_training();
    }
}

test "neural network 2 neuron layer" {
    var nn = try Network().init(std.testing.allocator, 1, @constCast(&[_]usize{ 2, 1 }));
    defer nn.deinit(std.testing.allocator);
    nn.learning_rate = 1e-1;

    // 2x + 1 = y
    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 1 });
    defer x.deinit(std.testing.allocator);
    x.buf[0] = 5;

    nn.layers[0].weights.buf[0] = 0;
    nn.layers[0].weights.buf[1] = 0;
    nn.layers[1].weights.buf[0] = 0;

    var i: usize = 0;
    var y = try nn.feed_forward(x, 11);
    nn.use_training();
    while (i < 1e2) : (i += 1) {
        std.log.warn("y: {d}\tw1: {any}\tw2: {d}", .{ y.buf[0], nn.layers[0].weights.buf, nn.layers[1].weights.buf });
        y = try nn.feed_forward(x, 11);
        nn.use_training();
    }
}

test "neural network 10-4-2 neuron layer" {
    var nn = try Network().init(std.testing.allocator, 1, @constCast(&[_]usize{ 10, 4, 2, 1 }));
    defer nn.deinit(std.testing.allocator);
    nn.learning_rate = 1e-4;

    // 2x + 1 = y
    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 1 });
    defer x.deinit(std.testing.allocator);
    x.buf[0] = 5;

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    nn.randomize(rand);

    var i: usize = 0;
    var y = try nn.feed_forward(x, 11);
    nn.use_training();
    while (i < 1e2) : (i += 1) {
        std.log.warn("y: {d}", .{y.buf[0]});
        y = try nn.feed_forward(x, 11);
        nn.use_training();
    }
}

test "neural network 10-4-2 neuron layer multiple inputs" {
    var nn = try Network().init(std.testing.allocator, 3, @constCast(&[_]usize{ 10, 4, 2, 1 }));
    defer nn.deinit(std.testing.allocator);
    nn.learning_rate = 1e-4;

    // x = 2
    // ax^2 + bx+ c = y
    const x = try mat.Matrix(f32, 2).init_alloc(std.testing.allocator, [2]usize{ 1, 3 });
    defer x.deinit(std.testing.allocator);
    x.buf[0] = 3;
    x.buf[1] = 6;
    x.buf[2] = 9;

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    nn.randomize(rand);

    var i: usize = 0;
    var y = try nn.feed_forward(x, 33);
    nn.use_training();
    while (i < 1e2) : (i += 1) {
        std.log.warn("y: {d}", .{y.buf[0]});
        y = try nn.feed_forward(x, 33);
        nn.use_training();
    }
}
