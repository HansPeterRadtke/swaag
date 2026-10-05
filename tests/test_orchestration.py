import pytest

from swaag.orchestration import OrchestrationStore


def test_durable_plan_graph_tracks_dependencies_and_policy(tmp_path):
    store = OrchestrationStore(tmp_path)
    plan = store.create_plan('Build and verify the release')
    build = store.add_node(plan.plan_id, 'Build release', worker_id='worker-build', priority=2.0, model_key='strong', finish_criteria='artifact built')
    verify = store.add_node(plan.plan_id, 'Verify release', worker_id='worker-verify', priority=1.0, model_key='default', abort_criteria='artifact missing')
    store.add_dependency(plan.plan_id, build, verify, condition={'when':'completed'}, input_mapping={'result':'release_artifact'})
    snap = store.snapshot(plan.plan_id)
    assert snap['plan'].revision == 4
    assert [n['worker_id'] for n in snap['nodes']] == ['worker-build','worker-verify']
    assert snap['edges'][0]['condition'] == {'when':'completed'}
    assert snap['edges'][0]['input_mapping'] == {'result':'release_artifact'}
    assert [e['event_type'] for e in snap['events']] == ['plan_created','node_added','node_added','dependency_added']


def test_plan_rejects_invalid_dependencies(tmp_path):
    store=OrchestrationStore(tmp_path)
    plan=store.create_plan('x')
    node=store.add_node(plan.plan_id,'one')
    try:
        store.add_dependency(plan.plan_id,node,node)
    except ValueError as exc:
        assert 'cannot target itself' in str(exc)
    else:
        raise AssertionError('self dependency accepted')


def test_runnable_nodes_follow_mechanical_dependencies_and_priority(tmp_path):
    store=OrchestrationStore(tmp_path)
    plan=store.create_plan('pipeline')
    first=store.add_node(plan.plan_id,'first',priority=1)
    second=store.add_node(plan.plan_id,'second',priority=3)
    third=store.add_node(plan.plan_id,'third',priority=2)
    store.add_dependency(plan.plan_id,first,third)
    assert [n['node_id'] for n in store.runnable_nodes(plan.plan_id)] == [second,first]
    store.set_node_state(plan.plan_id,first,'completed',worker_id='worker-first')
    assert [n['node_id'] for n in store.runnable_nodes(plan.plan_id)] == [second,third]


def test_unknown_semantic_condition_stays_blocked_until_orchestrator_decides(tmp_path):
    store=OrchestrationStore(tmp_path)
    plan=store.create_plan('conditional')
    source=store.add_node(plan.plan_id,'inspect')
    target=store.add_node(plan.plan_id,'repair')
    store.add_dependency(plan.plan_id,source,target,condition={'when':'semantic','question':'is repair needed?'})
    store.set_node_state(plan.plan_id,source,'completed')
    assert [n['node_id'] for n in store.runnable_nodes(plan.plan_id)] == []
    store.record_notification(plan.plan_id,'decision_required',{'node_id':target})
    assert store.snapshot(plan.plan_id)['events'][-1]['payload']['kind'] == 'decision_required'


def test_dependency_cycle_is_rejected(tmp_path):
    store=OrchestrationStore(tmp_path)
    plan=store.create_plan('cycle guard')
    a=store.add_node(plan.plan_id,'a')
    b=store.add_node(plan.plan_id,'b')
    c=store.add_node(plan.plan_id,'c')
    store.add_dependency(plan.plan_id,a,b)
    store.add_dependency(plan.plan_id,b,c)
    try:
        store.add_dependency(plan.plan_id,c,a)
    except ValueError as exc:
        assert 'cycle' in str(exc)
    else:
        raise AssertionError('cyclic plan accepted')


def test_node_revision_and_dependency_removal_are_durable(tmp_path):
    store=OrchestrationStore(tmp_path)
    plan=store.create_plan('mutable')
    a=store.add_node(plan.plan_id,'a')
    b=store.add_node(plan.plan_id,'b')
    edge=store.add_dependency(plan.plan_id,a,b)
    store.update_node(plan.plan_id,b,objective='b revised',priority=4.0,model_key='special')
    snap=store.snapshot(plan.plan_id)
    revised=next(n for n in snap['nodes'] if n['node_id']==b)
    assert revised['objective']=='b revised'
    assert revised['priority']==4.0
    assert revised['model_key']=='special'
    store.remove_dependency(plan.plan_id,edge)
    snap=store.snapshot(plan.plan_id)
    assert snap['edges']==[]
    assert snap['events'][-1]['event_type']=='dependency_removed'


def test_plan_policy_is_durable_and_validated(tmp_path):
    store = OrchestrationStore(tmp_path)
    plan = store.create_plan("ordered", scheduling_mode="sequential")
    assert plan.scheduling_mode == "sequential"
    assert plan.max_parallel == 0
    revised = store.set_plan_policy(
        plan.plan_id, scheduling_mode="parallel", max_parallel=2
    )
    assert revised.scheduling_mode == "parallel"
    assert revised.max_parallel == 2
    assert store.snapshot(plan.plan_id)["events"][-1]["event_type"] == "plan_policy_changed"


def test_semantic_dependency_requires_explicit_resolution(tmp_path):
    store = OrchestrationStore(tmp_path)
    plan = store.create_plan("branch")
    inspect = store.add_node(plan.plan_id, "inspect")
    repair = store.add_node(plan.plan_id, "repair")
    edge = store.add_dependency(
        plan.plan_id,
        inspect,
        repair,
        condition={"when": "semantic", "question": "repair needed?"},
    )
    store.set_node_state(plan.plan_id, inspect, "completed")
    assert repair not in [n["node_id"] for n in store.runnable_nodes(plan.plan_id)]
    store.resolve_dependency(
        plan.plan_id, edge, satisfied=True, decision="inspection found a defect"
    )
    assert repair in [n["node_id"] for n in store.runnable_nodes(plan.plan_id)]
    event = store.snapshot(plan.plan_id)["events"][-1]
    assert event["event_type"] == "dependency_resolved"
    assert event["payload"]["condition"]["satisfied"] is True


def test_orchestration_tool_schema_required_fields_match_properties_exactly():
    from swaag.tools.orchestration import OrchestrationControlTool

    schema = OrchestrationControlTool.input_schema
    assert len(schema["required"]) == len(set(schema["required"]))
    assert set(schema["required"]) == set(schema["properties"])
    assert "notifications.wait" in schema["properties"]["operation"]["enum"]


def test_orchestration_tool_validates_notification_wait_contract():
    from swaag.tools.orchestration import OrchestrationControlTool

    tool = OrchestrationControlTool()
    payload = {key: None for key in tool.input_schema["required"]}
    payload.update(
        {
            "operation": "notifications.wait",
            "plan_id": "plan-1",
            "after_sequence": 7,
            "unacknowledged_only": True,
            "timeout_seconds": 0.25,
        }
    )
    validated = tool.validate(payload)
    assert validated["after_sequence"] == 7
    assert validated["unacknowledged_only"] is True
    assert validated["timeout_seconds"] == 0.25
    assert tool.effective_kind(validated) == "pure"


def test_orchestration_bulk_plan_schema_is_portable_and_closed():
    from swaag.schema_portability import assert_portable_json_schema
    from swaag.tools.orchestration import OrchestrationControlTool

    assert_portable_json_schema(
        OrchestrationControlTool.input_schema,
        schema_name="orchestration_control",
    )
    plan_spec = OrchestrationControlTool.input_schema["properties"]["plan_spec"]
    object_variant = plan_spec["anyOf"][0]
    assert set(object_variant["properties"]) == set(object_variant["required"])
    node = object_variant["properties"]["nodes"]["items"]
    edge = object_variant["properties"]["edges"]["items"]
    assert set(node["properties"]) == set(node["required"])
    assert set(edge["properties"]) == set(edge["required"])
    assert "dependencies" not in object_variant["properties"]
    assert "node_id" not in node["properties"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1, True, "2"])
def test_invalid_node_priority_cannot_change_durable_plan(tmp_path, value):
    store = OrchestrationStore(tmp_path)
    plan = store.create_plan("validate priority")
    node = store.add_node(plan.plan_id, "valid node")
    before = store.snapshot(plan.plan_id)
    with pytest.raises(ValueError, match="finite and positive"):
        store.add_node(plan.plan_id, "invalid node", priority=value)
    with pytest.raises(ValueError, match="finite and positive"):
        store.update_node(plan.plan_id, node, priority=value)
    assert store.snapshot(plan.plan_id) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_resource_limits_cannot_replace_valid_limits(tmp_path, value):
    store = OrchestrationStore(tmp_path)
    plan = store.create_plan("resource admission")
    store.set_resource_limits(plan.plan_id, {"gpu": 1})
    before = store.snapshot(plan.plan_id)
    with pytest.raises(ValueError, match="finite and non-negative"):
        store.set_resource_limits(plan.plan_id, {"gpu": value})
    assert store.snapshot(plan.plan_id) == before


def test_orchestration_semantic_text_preserves_outer_whitespace_verbatim(tmp_path):
    store = OrchestrationStore(tmp_path)
    plan_objective = "\n  preserve plan objective exactly  \n"
    node_objective = "\n  preserve node objective exactly  \n"
    finish = "  exact finish criterion  \n"
    abort = "\n  exact abort criterion  "
    semantic_question = "\n  exact semantic question?  \n"

    plan = store.create_plan(plan_objective)
    source = store.add_node(
        plan.plan_id,
        node_objective,
        finish_criteria=finish,
        abort_criteria=abort,
    )
    target = store.add_node(plan.plan_id, "target")
    store.add_dependency(
        plan.plan_id,
        source,
        target,
        condition={"when": "semantic", "question": semantic_question},
    )

    snap = store.snapshot(plan.plan_id)
    source_node = next(item for item in snap["nodes"] if item["node_id"] == source)
    assert snap["plan"].objective == plan_objective
    assert source_node["objective"] == node_objective
    assert source_node["finish_criteria"] == finish
    assert source_node["abort_criteria"] == abort
    assert snap["edges"][0]["condition"]["question"] == semantic_question

    revised = "\n  revised exact objective  \n"
    revised_finish = "\n revised finish  "
    revised_abort = "  revised abort \n"
    store.update_node(
        plan.plan_id,
        source,
        objective=revised,
        finish_criteria=revised_finish,
        abort_criteria=revised_abort,
    )
    source_node = next(
        item for item in store.snapshot(plan.plan_id)["nodes"]
        if item["node_id"] == source
    )
    assert source_node["objective"] == revised
    assert source_node["finish_criteria"] == revised_finish
    assert source_node["abort_criteria"] == revised_abort


def test_orchestration_tool_plan_validation_preserves_semantic_text():
    from swaag.tools.orchestration import OrchestrationControlTool

    tool = OrchestrationControlTool()
    payload = {key: None for key in tool.input_schema["required"]}
    payload.update(
        {
            "operation": "plan.apply",
            "plan_spec": {
                "objective": "\n plan objective \n",
                "scheduling_mode": "parallel",
                "max_parallel": 0,
                "reporting_mode": "important",
                "resource_limits_json": None,
                "nodes": [
                    {
                        "key": " node-key ",
                        "completion_mode": "natural",
                        "objective": "\n node objective \n",
                        "priority": 1.0,
                        "model_key": " default ",
                        "finish_criteria": " finish exactly \n",
                        "abort_criteria": "\n abort exactly ",
                        "resources_json": None,
                    }
                ],
                "edges": [
                    {
                        "source": " node-key ",
                        "target": " target-key ",
                        "when": "semantic",
                        "semantic_question": "\n semantic question exactly? \n",
                        "input_mapping_json": None,
                    }
                ],
                "start": False,
            },
        }
    )
    # Validation is independent of graph endpoint existence; materialization checks that.
    validated = tool.validate(payload)
    spec = validated["plan_spec"]
    assert spec["objective"] == "\n plan objective \n"
    assert spec["nodes"][0]["key"] == "node-key"
    assert spec["nodes"][0]["objective"] == "\n node objective \n"
    assert spec["nodes"][0]["model_key"] == "default"
    assert spec["nodes"][0]["finish_criteria"] == " finish exactly \n"
    assert spec["nodes"][0]["abort_criteria"] == "\n abort exactly "
    assert spec["edges"][0]["source"] == "node-key"
    assert spec["edges"][0]["target"] == "target-key"
    assert spec["edges"][0]["condition"]["question"] == "\n semantic question exactly? \n"


def test_orchestration_tool_preserves_direct_semantic_fields_and_normalizes_ids():
    from swaag.tools.orchestration import OrchestrationControlTool

    tool = OrchestrationControlTool()
    payload = {key: None for key in tool.input_schema["required"]}
    payload.update(
        {
            "operation": "node.revise",
            "plan_id": " plan-id ",
            "node_id": " node-id ",
            "objective": "\n exact objective \n",
            "decision": " exact decision \n",
            "model_key": " strong ",
            "finish_criteria": "\n exact finish ",
            "abort_criteria": " exact abort \n",
            "reason": "\n exact reason \n",
        }
    )

    validated = tool.validate(payload)

    assert validated["plan_id"] == "plan-id"
    assert validated["node_id"] == "node-id"
    assert validated["model_key"] == "strong"
    assert validated["objective"] == "\n exact objective \n"
    assert validated["decision"] == " exact decision \n"
    assert validated["finish_criteria"] == "\n exact finish "
    assert validated["abort_criteria"] == " exact abort \n"
    assert validated["reason"] == "\n exact reason \n"
