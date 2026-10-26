import hashlib,json
from pathlib import Path
BASE=Path('/data/var/perceptual-independent-20261024/final_encoded_owner_mp4_20261026')
DOC=Path(__file__).resolve().parents[2]/'docs/research/perceptual-owner-final-mp4-playback-2026-10-26.md'
JSON=DOC.with_suffix('.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if DOC.exists() or JSON.exists():raise RuntimeError('Published artifacts exist; no rewrite')
 f=json.loads((BASE/'source_manifest.json').read_text());p=json.loads((BASE/'decoded_frame_evidence.json').read_text());b=json.loads((BASE/'browser_html5_playback_20261026.json').read_text())
 assert p['source_manifest_sha256']==sha(BASE/'source_manifest.json')
 assert p['protocol_sha256']==sha(BASE/'protocol.json')
 assert b['decoded_video_source_fidelity_sha256']==sha(BASE/'decoded_frame_evidence.json')
 assert p['source_matched_frame_sequence_property']=='PASS' and b['status']=='PASS'
 assert len(f['original_source_PNGs'])==210 and len(p['all_encoded_native_frames']['warehouse'])==210
 report={'date':'2026-10-26','status':'PARTIAL','source':{'scene_sha256':f['source_scene_sha256'],'render_status_sha256':f['render_status_sha256'],'original_frame_count':210,'all_source_frame_hashes_verified':True},'mp4_sha256':p['video_sha256'],'negative_dock_sha256':p['dock_negative_sha256'],'pts':p['PTS'],'decoded':{'source_matched_frame_sequence_property':p['source_matched_frame_sequence_property'],**p['summary']},'tail':p['tail'],'browser':{k:v for k,v in b.items() if k!='sampled_states'},'latency':p['latency'],'audio_synchronization':'REVIEW','subjective_video_quality':'REVIEW','physical_contact_from_video':'REVIEW','gpu_used':False,'original_mp4s_unmodified':True,'storyboard_transfer_denied':True,'evidence_sha256':{x:sha(BASE/x) for x in ('protocol.json','source_manifest.json','decoded_frame_evidence.json','browser_html5_playback_20261026.json','actual_final_warehouse_mp4_decoded_contact_sheet.jpg','actual_browser_playing_warehouse_film.png')},'limits':['Encoded frames match the author-rendered source but cannot establish photorealism or semantic 3D geometry.','The warehouse original contains no audio stream; sync cannot be tested.','Different dock and shifted source are invalid reference negatives, not deliberately faulty films.','Firefox headless file playback reaching ended is a playback smoke test, not the user watching/approving video.','Transfer of the contact sheet into chat was denied, not repeated; local original retained.']}
 JSON.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 w=p['summary']['warehouse'];dock=p['summary']['dock'];tail=p['tail']
 DOC.write_text(f'''# Final encoded owner warehouse video — actual output playback verification (26 October 2026)

**Result: PASS for the narrowly specified encoded-source identity, frame fidelity, presentation timestamps and headless Firefox HTML5 playback. Overall visual research remains PARTIAL.** Inspected and SHA-verified all 210 original 960×540 Cycles PNG files from source scene `{f['source_scene_sha256']}`. Verified the final approved 14-second 15-fps 210-frame warehouse MP4 by SHA `{p['video_sha256']}`; compared all decoded RGB video frames to corresponding original PNG. This is actual final encoded media, not just metadata/JSON. The source manifests and immutable videos were never modified.

| Measured production-adjacent property | Actual result |
|---|---|
| Total decoded final frames | 210/210 |
| PTS nonmonotonic frames | 0; maximum error {p['PTS']['warehouse']['frame_timebase_max_error_seconds']:.8f} seconds |
| Source-aligned native decoded PSNR | min **{w['source_alignment_PSN_dB_min']:.2f} dB**, median **{w['source_alignment_PSN_dB_median']:.2f} dB** |
| Aligned SSIM grayscale / 240px | median **{w['SSIM_gray240_median']:.5f}** |
| Deliberate different approved dock video against wrong warehouse frames | median **{dock['source_alignment_PSN_dB_median']:.2f} dB** |
| Wrong warehouse source shifted 15 encoded frames (1 s) | median **{w['shifted_source_15frame_PSN_dB_median']:.2f} dB** |
| Source camera final 3-second stop, output adjacent-pixel motion proxy | early median **{tail['early_first_165_decoded_adjacent_frame_gray_motion_median']:.5f}**, tail median **{tail['tail_last_45_decoded_adjacent_frame_gray_motion_median']:.5f}** |
| Real Firefox headless native HTML5 player | video loaded at 960×540, play() successful, currentTime **14.0 seconds**, `ended=true`, `error=null` |

Two external-to-video negative reference controls confirm that the high fidelity is not merely a generic visually plausible stream. They are valid independent videos/frames that fail the *wrong reference match*, not semantic negative examples of bad video quality. Browser test ran an isolated Firefox instance, not a production browser profile, and saved a screenshot during playback at media time {b['screenshots_saved'][0]['time_s']:.3f} s. Both decoded contact sheet and browser screenshot remain in isolated `/data/var/perceptual-independent-20261024/final_encoded_owner_mp4_20261026/` storage. The operation to transfer the contact sheet into this chat was denied, and was not retried or routed around. No original media, publishing target, source blend, renderer, active GPU, gallery, earlier uncertain effects or other workers' state was changed.

**Acceptance boundary:** The warehouse's scene/source-specific static mesh support contact has separate prior 72/72 evidence, but the encoded movie does not independently certify mechanical contact, lighting realism or aesthetic quality. The source intentionally includes a stationary three-second camera hold; this is not a video freeze defect. The actual MP4 has no audio stream, so audio event synchronization is untestable and REVIEW. Firefox playback is a real final-output smoke test, not a human perceptual approval. The old high-resolution v52 retrieval denial was not revisited. General GUI, arrow topology, full 3D semantic completeness and unexpected natural video fault classification remain unqualified. **Overall PARTIAL.**

Reproduction: `python3 tests/research/owner_final_encoded_mp4_acceptance_20261026.py freeze|decode` then `python3 tests/research/owner_final_html5_playback_20261026.py`, then `python3 tests/research/publish_owner_final_mp4_acceptance_20261026.py`; stages refuse overwriting their preserved original receipts. Offline unit checks test the committed compact summary. Native source PNG SHA and all decoded per-frame measures remain local, not Git.
''')
 print('REPORT',JSON,'sha',sha(JSON))
if __name__=='__main__':main()
