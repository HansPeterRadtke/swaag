"""In-sample diagnostic tests; these are not model-quality evidence."""
from pathlib import Path
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/demos'))
try:
    import diagnose_natural_turn_zero_false_frontier as f
finally:
    sys.path.pop(0)


def test_threshold_equality_cannot_pass_zero_false_screen():
    assert f.no_hold_threshold([0.7])>0.7
    assert f.no_hold_threshold([1.0])>1.0
    assert f.no_hold_threshold([])==0.0


def test_best_conjunction_can_use_complementary_signals_not_missing_ones():
    rows=[
        dict(label='hold',audio=.9,text=.1),
        dict(label='hold',audio=.1,text=.9),
        dict(label='eot',audio=.8,text=.8),
        dict(label='eot',audio=None,text=.9),
        dict(label='eot',audio=.9,text=None),
    ]
    a=f.best_single(rows,'audio')
    t=f.best_single(rows,'text')
    b=f.best_and(rows)
    assert a['eot_detected']==t['eot_detected']==0
    assert b['eot_detected']==1 and b['eot_total']==3
    assert b['false_holds']==0


def test_perfect_false_score_at_one_remains_excluded():
    rows=[dict(label='hold',audio=1.0,text=1.0),
          dict(label='eot',audio=1.0,text=1.0)]
    assert f.best_single(rows,'audio')['eot_detected']==0
    assert f.best_and(rows)['eot_detected']==0


def test_frontier_matches_brute_force_on_small_grid():
    rows=[
        dict(label='hold',audio=.2,text=.6),
        dict(label='hold',audio=.7,text=.3),
        dict(label='hold',audio=.4,text=.4),
        dict(label='eot',audio=.8,text=.5),
        dict(label='eot',audio=.5,text=.7),
        dict(label='eot',audio=.9,text=.2),
    ]
    candidates={0.0,1.1}
    for r in rows:
        candidates.update((r['audio'],r['text'],math.nextafter(r['audio'],math.inf),
                           math.nextafter(r['text'],math.inf)))
    brute=max(sum(r['label']=='eot' and r['audio']>=a and r['text']>=t for r in rows)
              for a in candidates for t in candidates
              if not any(r['label']=='hold' and r['audio']>=a and r['text']>=t for r in rows))
    assert f.best_and(rows)['eot_detected']==brute
