"""Testing SetsEncoder"""
import pytest
import torch
from fdsa.models.encoders.encoder_sets_ae import EncoderSetsAE


@pytest.fixture
def params():
    batch_size = 5
    input_size = 10
    sequence_length = 8
    hidden_sizes_linear = 20
    hidden_sizes_encoder = 20
    cell = 'LSTM'
    return {
        'cell': cell,
        'batch_size': batch_size,
        'input_size': input_size,
        'sequence_length': sequence_length,
        'hidden_size_linear': hidden_sizes_linear,
        'hidden_size_encoder': hidden_sizes_encoder
    }


def test_memory_mapping(params):
    """Test linear mapping to memory locations."""

    input_set = torch.rand(
        (
            params['batch_size'], params['sequence_length'],
            params['input_size']
        )
    )
    set_ae = EncoderSetsAE(**params)

    with torch.no_grad():
        memory_slots = set_ae.memory_mapping(input_set)
        assert memory_slots.size() == torch.Size(
            [
                params['batch_size'], params['sequence_length'],
                params['hidden_size_linear']
            ]
        )


def test_set_ae(params):
    """Test dimension correctness of the encoder outputs."""

    input_set = torch.rand(
        (
            params['batch_size'], params['sequence_length'],
            params['input_size']
        )
    )
    set_ae = EncoderSetsAE(**params)

    cell_state, hidden_state, read_vector = set_ae(input_set)

    assert cell_state.size() == torch.Size(
        [params['batch_size'], params['hidden_size_encoder']]
    )
    assert hidden_state.size() == torch.Size(
        [params['batch_size'], params['hidden_size_encoder']]
    )

    assert read_vector.size() == torch.Size(
        [params['batch_size'], params['hidden_size_encoder']]
    )
