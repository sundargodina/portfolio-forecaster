# dashboard/test_app.py
import pytest
import dash
from app import manage_portfolio_state
from dash.dash import no_update

# Mock the dash.callback_context object
class MockContext:
    def __init__(self, triggered_prop_id):
        self.triggered = [{'prop_id': triggered_prop_id}] if triggered_prop_id else []
        self.triggered_id = triggered_prop_id
        self.states = {}
        self.inputs = {}
        self.outputs = {}

# Test for adding an asset
def test_add_asset():
    """
    Simulates adding 'SampleAsset' and checks if the state is updated correctly.
    """
    # Simulate callback inputs and states
    add_clicks = 1
    remove_clicks = [no_update] # Changed from `[None]` to avoid a bug in a multi-input scenario
    selected_asset = 'SampleAsset'
    current_state = []
    current_weights = {}

    # Create a mock context for the 'add-asset-btn' click
    mock_ctx = MockContext('add-asset-btn.n_clicks')

    # Call the callback function directly, passing the mock context
    new_state, new_weights = manage_portfolio_state(
        add_clicks, remove_clicks, selected_asset, current_state, current_weights, mock_ctx
    )

    # Assert the expected output
    assert new_state == ['SampleAsset']
    assert new_weights == {'SampleAsset': 0}
    print("Test for adding a new asset passed! ")

# Test for removing an asset
def test_remove_asset():
    """
    Simulates removing 'ExistingAsset' and checks if the state is updated correctly.
    """
    # Simulate callback inputs and states
    add_clicks = 0
    remove_clicks = [{'type': 'remove-btn', 'asset_name': 'ExistingAsset'}]
    selected_asset = None
    current_state = ['ExistingAsset', 'AnotherAsset']
    current_weights = {'ExistingAsset': 0.5, 'AnotherAsset': 0.5}
    
    # Create a mock context for the 'remove-btn' click
    mock_ctx = MockContext('{"asset_name":"ExistingAsset","type":"remove-btn"}.n_clicks')

    # Call the callback function directly, passing the mock context
    new_state, new_weights = manage_portfolio_state(
        add_clicks, remove_clicks, selected_asset, current_state, current_weights, mock_ctx
    )
    
    # Assert the expected output
    assert new_state == ['AnotherAsset']
    assert new_weights == {'AnotherAsset': 0.5}
    print("Test for removing an asset passed! ")
    
    
    
    
    def test_update_weights():
    """
    Checks if the weights store is updated with a new value.
    """
    # Simulate a user typing a value (e.g., 0.6) into a weight input
    weights_values = [0.6]
    weights_ids = [{'type': 'weight-input', 'asset_name': 'SampleAsset'}]
    weights_data = {} # Initial state of the store

    # Call the callback directly
    new_weights_data = app.update_weights_store(weights_values, weights_ids, weights_data)

    # Assert that the new value is correctly stored
    assert new_weights_data['SampleAsset'] == 0.6
    print("Test for updating weights passed! ")
    
    def test_normalization_message():
    """
    Checks if the normalization message appears and disappears correctly.
    """
    # Scenario 1: Weights do not sum to 1
    weights_values_fail = [0.4, 0.4]
    weights_data_fail = {'AssetA': 0.4, 'AssetB': 0.4}
    message_fail = app.check_weights_and_store_message(weights_values_fail, weights_data_fail)
    assert "Note: Your allocations do not sum to 1" in message_fail
    print("Normalization message test (fail) passed! ")

    # Scenario 2: Weights sum to 1
    weights_values_pass = [0.5, 0.5]
    weights_data_pass = {'AssetA': 0.5, 'AssetB': 0.5}
    message_pass = app.check_weights_and_store_message(weights_values_pass, weights_data_pass)
    assert message_pass == ""
    print("Normalization message test (pass) passed! ")