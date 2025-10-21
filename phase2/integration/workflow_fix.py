# Fix script for workflow integration
import re

with open('phase2/integration/workflow_integration.py', 'r') as f:
    content = f.read()

# Fix method names
content = content.replace('start_capture(', 'start_trajectory(')
content = content.replace('image_data=image_data', 'initial_state=image_data')
content = content.replace('complete_capture(', 'finalize_trajectory(')
content = content.replace('final_output=', 'final_state=')
content = content.replace('output_data=assessment_results', 'final_state=assessment_results')

with open('phase2/integration/workflow_integration.py', 'w') as f:
    f.write(content)

print("✅ Fixed workflow integration methods")
