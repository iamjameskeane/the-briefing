
import asyncio
import os
from unittest.mock import MagicMock, patch, AsyncMock
from agents.editor import run_editor_agent
from agents.editor import EditorInput
from tools import get_memory_tools

# Mock config
os.environ["GEMINI_API_KEY"] = "test_key"
os.environ["TAVILY_API_KEY"] = "test_key"

async def test_editor_memory_tool():
    print("Testing Editor Memory Tool Integration...")
    
    # Mock input data
    input_data = EditorInput(
        thematic_clusters=[],
        hub_candidates=[],
        total_word_budget=1000,
        previous_edition={
            "run_id": "briefing_20260125_110850",
            "date": "2026-01-25",
            "notes": "Edition 1"
        }
    )

    # Mock Gemini client and response
    with patch("agents.editor.genai.Client") as MockClient:
        mock_client = MockClient.return_value
        mock_model = mock_client.aio.models
        
        # 1. Tool Call (read_past_briefing)
        mock_response_tool = MagicMock()
        mock_part_tool = MagicMock()
        mock_part_tool.function_call.name = "read_past_briefing"
        mock_part_tool.function_call.args = {"run_id": "briefing_20260125_110850"}
        mock_response_tool.candidates = [MagicMock(content=MagicMock(parts=[mock_part_tool]))]
        
        # 2. Final Text Response
        mock_response_final = MagicMock()
        mock_response_final.text = "This is the editorial brief with historical context."
        
        # Setup candidates for the final response to satisfy run_editor_agent checks
        final_part = MagicMock()
        final_part.text = "This is the editorial brief with historical context."
        final_part.function_call = None
        mock_response_final.candidates = [MagicMock(content=MagicMock(parts=[final_part]), finish_reason="STOP")]
        
        # Set side effects
        mock_model.generate_content = AsyncMock(side_effect=[mock_response_tool, mock_response_final])
        
        # Mock execute_tool_call
        with patch("agents.editor.execute_tool_call") as mock_execute:
            mock_execute.return_value = "Historical data for Edition 1"
            
            try:
                await run_editor_agent(input_data, client=mock_client)
                print("✅ Editor successfully called memory tools.")
                
                # Verify tool call
                mock_execute.assert_called_with("read_past_briefing", {"run_id": "briefing_20260125_110850"})
                print("✅ Verified call to 'read_past_briefing'")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_editor_memory_tool())
