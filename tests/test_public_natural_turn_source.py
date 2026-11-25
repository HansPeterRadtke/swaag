from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    from verify_public_natural_turn_source import gold_prefix
finally:
    sys.path.pop(0)

def test_causal_prefix_stops_at_first_invalid_or_future_word():
    w=[{'start':0.0,'end':.1,'word':'yes'},
       {'start':.2,'end':.7,'word':'later'},
       {'start':0.0,'end':.1,'word':'not reached'}]
    assert gold_prefix(w,3200)=='yes'
    w[1]['end']=float('nan')
    assert gold_prefix(w,16000)=='yes'
    assert gold_prefix(w,0)==''
