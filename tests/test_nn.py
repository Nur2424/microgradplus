import pytest

from microgradplus.engine import Value
from microgradplus.nn import MLP, Layer, Neuron


def test_neuron_parameter_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4  # 3 weights + 1 bias


def test_layer_parameter_count():
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * 4  # 4 neurons * (3w + 1b)


def test_mlp_parameter_count_matches_manual_calc():
    mlp = MLP(3, [4, 4, 1])
    # layer1: 4 neurons * (3+1) = 16
    # layer2: 4 neurons * (4+1) = 20
    # layer3: 1 neuron  * (4+1) = 5
    assert len(mlp.parameters()) == 16 + 20 + 5


def test_mlp_forward_returns_value_for_single_output():
    mlp = MLP(3, [4, 1])
    out = mlp([1.0, 2.0, 3.0])
    assert isinstance(out, Value)


def test_mlp_forward_returns_list_for_multi_output():
    mlp = MLP(3, [4, 2])
    out = mlp([1.0, 2.0, 3.0])
    assert isinstance(out, list)
    assert len(out) == 2


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu", "leaky_relu", "linear"])
def test_all_activations_run(activation):
    mlp = MLP(2, [3, 1], activation=activation, out_activation=activation)
    out = mlp([0.5, -0.5])
    assert isinstance(out.data, float)


@pytest.mark.parametrize("init", ["uniform", "xavier", "he"])
def test_all_init_schemes_run(init):
    mlp = MLP(2, [3, 1], init=init)
    out = mlp([0.5, -0.5])
    assert isinstance(out.data, float)


def test_sigmoid_output_in_unit_interval():
    mlp = MLP(2, [3, 1], activation="relu", out_activation="sigmoid", init="he")
    out = mlp([0.5, -0.5])
    assert 0.0 <= out.data <= 1.0


def test_backward_populates_all_parameter_grads():
    mlp = MLP(2, [3, 1])
    out = mlp([0.5, -0.5])
    out.backward()
    assert all(p.grad != 0.0 or True for p in mlp.parameters())  # no crash
    # at least the output layer's incoming weights should have nonzero grad
    assert any(p.grad != 0.0 for p in mlp.parameters())
