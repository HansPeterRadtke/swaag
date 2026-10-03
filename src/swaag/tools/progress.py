from swaag.progress import validate_progress, step_percentage
from swaag.tools.base import Tool, ToolValidationError
from swaag.types import ToolExecutionResult, ToolGeneratedEvent
from swaag.utils import stable_json_dumps


class ReportProgressTool(Tool):
    name = 'report_progress'
    description = 'Record an evidence-based work breakdown, rough progress, and optionally a qualified time estimate.'
    usage_guidance = ('Use meaningful explicit steps and relative effort weights. Mark a step completed only on evidence. '
        'Revise the breakdown when scope changes. Percentages describe these steps, not token generation or wall time. '
        'For continuous work report the current finite cycle; overall completion remains intentionally endless. '
        'Use null for unknown remaining time. Any estimate must state the model, input scale, machine, and load assumptions.')
    kind = 'stateful'
    input_schema = {'type': 'object', 'additionalProperties': False,
        'properties': {'steps': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
            'properties': {'id': {'type': 'string'}, 'label': {'type': 'string'},
                           'state': {'type': 'string', 'enum': ['pending', 'in_progress', 'completed']},
                           'weight': {'type': 'number'}}, 'required': ['id', 'label', 'state', 'weight']}},
            'reason': {'type': 'string'}, 'estimated_remaining_seconds': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
            'estimate_conditions': {'type': 'string'}},
        'required': ['steps', 'reason', 'estimated_remaining_seconds', 'estimate_conditions']}

    def validate(self, raw_input):
        try:
            return validate_progress(raw_input)
        except ValueError as exc:
            raise ToolValidationError(str(exc)) from exc

    def required_generated_event_types(self, validated_input):
        return {'agent_progress'}

    def execute(self, validated_input, context):
        if len(stable_json_dumps(validated_input)) > context.config.runtime.max_open_question_chars:
            raise ToolValidationError('Progress report exceeds configured state character budget')
        output = {**validated_input, 'step_percent': step_percentage(validated_input), 'assessment_source': 'model'}
        return ToolExecutionResult(self.name, output, stable_json_dumps(output),
            generated_events=[ToolGeneratedEvent('agent_progress', validated_input)])


PROGRESS_TOOLS = [ReportProgressTool()]
