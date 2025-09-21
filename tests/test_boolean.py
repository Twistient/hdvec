import numpy as np

from hdvec.encoding.boolean import (
    BooleanEncoder,
    apply_truth_table,
    logic_and,
    logic_and_vector,
    logic_not,
    logic_not_vector,
    logic_or,
    logic_or_vector,
    logic_xor,
    logic_xor_vector,
)


def test_boolean_encoder_roundtrip():
    encoder = BooleanEncoder(D=32, rng=np.random.default_rng(0))
    for bit in (0, 1):
        encoded = encoder.encode(bit)
        decoded = encoder.decode(encoded)
        assert decoded == bit


def test_logic_ops():
    encoder = BooleanEncoder(D=32, rng=np.random.default_rng(1))
    values = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a_bit, b_bit in values:
        a = encoder.encode(a_bit)
        b = encoder.encode(b_bit)
        assert encoder.decode(logic_and(a, b, encoder)) == (a_bit & b_bit)
        assert encoder.decode(logic_and_vector(a, b, encoder)) == (a_bit & b_bit)
        assert encoder.decode(logic_or(a, b, encoder)) == (a_bit | b_bit)
        assert encoder.decode(logic_or_vector(a, b, encoder)) == (a_bit | b_bit)
        assert encoder.decode(logic_xor(a, b, encoder)) == (a_bit ^ b_bit)
        assert encoder.decode(logic_xor_vector(a, b, encoder)) == (a_bit ^ b_bit)
        assert encoder.decode(logic_not(a, encoder)) == (1 - a_bit)
        assert encoder.decode(logic_not_vector(a, encoder)) == (1 - a_bit)


def test_apply_truth_table():
    table = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    assert apply_truth_table((1, 1), table) == 1
    assert apply_truth_table((0, 1), table) == 0
