
import asyncio
import os
from unittest.mock import MagicMock, patch, AsyncMock
from agents.analyst import run_analyst_agent
from agents.schemas import AnalystInput
from tools import get_graph_tools

# Mock config
os.environ["GEMINI_API_KEY"] = "test_key"
os.environ["TAVILY_API_KEY"] = "test_key"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_SERVICE_KEY"] = "test_key"

async def test_analyst_graph_tools():
    print("Testing Analyst Graph Tool Integration...")
    
    # Mock input data
    input_data = AnalystInput(
        cluster_id="test_cluster",
        theme_label="Test Theme",
        events=[
            {"id": "evt_123", "title": "Test Event", "severity": 8, "summary": "A major test event."}
        ],
        regions_touched=["Global"],
        is_featured=False
    )

    # Mock Gemini client and response
    with patch("agents.analyst.genai.Client") as MockClient:
        mock_client = MockClient.return_value
        mock_model = mock_client.aio.models
        
        # Mock response to trigger tool calls
        # 1. get_event_entities
        mock_response_tool_1 = MagicMock()
        mock_part_1 = MagicMock()
        mock_part_1.function_call.name = "get_event_entities"
        mock_part_1.function_call.args = {"event_id": "evt_123"}
        mock_response_tool_1.candidates = [MagicMock(content=MagicMock(parts=[mock_part_1]))]
        
        # 2. resolve_entity
        mock_response_tool_2 = MagicMock()
        mock_part_2 = MagicMock()
        mock_part_2.function_call.name = "resolve_entity"
        mock_part_2.function_call.args = {"name": "Test Actor"}
        mock_response_tool_2.candidates = [MagicMock(content=MagicMock(parts=[mock_part_2]))]

        # 3. get_entity_relationships
        mock_response_tool_3 = MagicMock()
        mock_part_3 = MagicMock()
        mock_part_3.function_call.name = "get_entity_relationships"
        mock_part_3.function_call.args = {"entity_name": "Test Actor"}
        mock_response_tool_3.candidates = [MagicMock(content=MagicMock(parts=[mock_part_3]))]
        
        # Mock response for final output
        mock_response_final = MagicMock()
        mock_response_final.text = """
        {
            "region": "Global",
            "geopolitical_archetype": "Test Archetype",
            "archetype_explanation": "Explanation",
            "primary_actors": [],
            "futures_wheel": {
                "driver_event": "Test Event",
                "driver_event_id": "evt_123",
                "first_order": "Effect 1",
                "second_order": "Effect 2",
                "third_order": "Effect 3"
            },
            "competing_hypotheses": {
                "consensus": "H1",
                "contrarian": "H2",
                "contradicting_evidence": [],
                "evidence_event_ids": []
            },
            "pmesii_tags": {},
            "confidence": "HIGH",
            "confidence_rationale": "Rationale",
            "external_sources": []
        }
        """
        
        # Mock response after tool execution (intermediate reasoning)
        mock_response_text = MagicMock()
        text_part = MagicMock(text="Based on the graph...")
        text_part.function_call = None  # Crucial: explicit None to avoid MagicMock creating one
        mock_response_text.content.parts = [text_part]
        mock_response_text.candidates = [MagicMock(content=mock_response_text.content)]

        # Set side effects for generate_content using AsyncMock
        # 1. get_event_entities -> 2. resolve_entity -> 3. get_entity_relationships -> 4. Reasoning -> 5. Final JSON
        mock_model.generate_content = AsyncMock(side_effect=[
            mock_response_tool_1, 
            mock_response_tool_2,
            mock_response_tool_3,
            mock_response_text, 
            mock_response_final
        ])
        
        # Mock execute_tool_call to avoid actual API calls
        with patch("agents.analyst.execute_tool_call") as mock_execute:
            mock_execute.return_value = "Graph data for evt_123"
            
            try:
                await run_analyst_agent(input_data, client=mock_client)
                print("✅ Analyst successfully called tools and produced output.")
                
                # Verify tool calls
                mock_execute.assert_any_call("get_event_entities", {"event_id": "evt_123"})
                mock_execute.assert_any_call("resolve_entity", {"name": "Test Actor"})
                mock_execute.assert_any_call("get_entity_relationships", {"entity_name": "Test Actor"})
                print("✅ Verified calls to 'get_event_entities', 'resolve_entity', and 'get_entity_relationships'")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_analyst_graph_tools())
