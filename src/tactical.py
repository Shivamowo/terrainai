"""
TerrainAI Tactical — mission analysis layer.
Consumes terrain_stats and zone_map produced by src/inference.py.
"""

import datetime
from src.inference import CLASSES, ABSENT_CLASS_IDS


def get_traversability_score(terrain_stats: dict) -> dict:
    """
    Compute a 0-100 traversability score from active, present classes.
    Higher = easier to cross.
    """
    per_class = terrain_stats['per_class']
    score = 0.0
    breakdown = {}

    for cls_id, cls_data in per_class.items():
        if cls_id in ABSENT_CLASS_IDS:
            continue
        if not CLASSES[cls_id]['present_in_dataset']:
            continue
        if not cls_data['active']:
            continue
        contribution = CLASSES[cls_id]['weight'] * cls_data['percentage']
        score += contribution
        breakdown[CLASSES[cls_id]['name']] = {
            'weight': CLASSES[cls_id]['weight'],
            'percentage': cls_data['percentage'],
            'contribution': round(contribution, 4),
        }

    score = round(min(max(score, 0.0), 100.0), 2)

    if score >= 70:
        rating = 'HIGH'
        recommendation = 'Terrain passable. Advance recommended.'
    elif score >= 40:
        rating = 'MEDIUM'
        recommendation = 'Partial obstruction. Proceed with caution.'
    else:
        rating = 'LOW'
        recommendation = 'Terrain heavily obstructed. Do not advance.'

    return {
        'score': score,
        'rating': rating,
        'recommendation': recommendation,
        'breakdown': breakdown,
    }


def get_threat_assessment(terrain_stats: dict, zone_map: list) -> dict:
    """
    Assess tactical threats from alert classes and impassable zones.
    Only processes present_in_dataset=True classes.
    """
    per_class = terrain_stats['per_class']
    traversability_score = 0.0
    # Recompute score inline to keep this function self-contained
    for cls_id, cls_data in per_class.items():
        if cls_id in ABSENT_CLASS_IDS or not CLASSES[cls_id]['present_in_dataset']:
            continue
        if cls_data['active']:
            traversability_score += CLASSES[cls_id]['weight'] * cls_data['percentage']

    alerts = []
    for cls_id, cls_data in per_class.items():
        if cls_id in ABSENT_CLASS_IDS:
            continue
        if not CLASSES[cls_id]['present_in_dataset']:
            continue
        if not (CLASSES[cls_id]['alert'] and cls_data['active'] and cls_data['percentage'] > 0.5):
            continue
        if cls_id == 8:
            message = 'OBSTACLE — potential cover or concealment detected'
        elif cls_id == 9:
            message = 'ANOMALY — unusual vegetation pattern detected'
        else:
            message = f'ALERT — {CLASSES[cls_id]["name"]} detected'
        alerts.append({
            'class_id': cls_id,
            'name': CLASSES[cls_id]['name'],
            'percentage': cls_data['percentage'],
            'message': message,
        })

    blocked_zones = []
    for row in zone_map:
        for zone in row:
            if zone['traversability_score'] < 20:
                blocked_zones.append({
                    'zone_row': zone['zone_row'],
                    'zone_col': zone['zone_col'],
                    'traversability_score': zone['traversability_score'],
                    'dominant_class': zone['dominant_class_name'],
                })

    has_alerts = len(alerts) > 0
    low_traversability = traversability_score < 40

    if has_alerts and low_traversability:
        threat_level = 'HIGH'
    elif has_alerts or low_traversability:
        threat_level = 'MEDIUM'
    else:
        threat_level = 'LOW'

    summary_parts = []
    if alerts:
        names = ', '.join(a['name'] for a in alerts)
        summary_parts.append(f"Active alerts: {names}.")
    if blocked_zones:
        summary_parts.append(f"{len(blocked_zones)} zone(s) blocked.")
    if not summary_parts:
        summary_parts.append("No active threats detected.")
    summary = ' '.join(summary_parts)

    return {
        'threat_level': threat_level,
        'alerts': alerts,
        'blocked_zones': blocked_zones,
        'summary': summary,
    }


def get_movement_recommendation(traversability: dict, threat: dict) -> dict:
    """
    Synthesise a movement recommendation from traversability and threat data.
    """
    trav_rating = traversability['rating']
    threat_level = threat['threat_level']

    if threat_level == 'HIGH' and trav_rating == 'LOW':
        primary_action = 'RETREAT'
    elif threat_level == 'HIGH' or trav_rating == 'LOW':
        primary_action = 'HOLD'
    elif threat_level == 'MEDIUM' or trav_rating == 'MEDIUM':
        primary_action = 'PROCEED WITH CAUTION'
    else:
        primary_action = 'ADVANCE'

    reasoning = (
        f"Traversability score is {traversability['score']:.1f}/100 ({trav_rating}), "
        f"indicating {traversability['recommendation'].lower()} "
        f"Threat assessment is {threat_level}: {threat['summary']}"
    )

    # Safe zones: high traversability, no active alerts
    alert_zone_coords = {(z['zone_row'], z['zone_col']) for z in threat.get('blocked_zones', [])}
    safe_zones = []
    avoid_zones = []

    for row_zones in _zone_map_from_threat(threat):
        for zone in row_zones:
            coord = (zone['zone_row'], zone['zone_col'])
            ts = zone['traversability_score']
            in_alert = coord in alert_zone_coords
            if ts > 60 and not in_alert:
                safe_zones.append({'row': zone['zone_row'], 'col': zone['zone_col'], 'score': ts})
            if ts < 30 or in_alert:
                avoid_zones.append({'row': zone['zone_row'], 'col': zone['zone_col'], 'score': ts})

    return {
        'primary_action': primary_action,
        'reasoning': reasoning,
        'safe_zones': safe_zones,
        'avoid_zones': avoid_zones,
    }


def _zone_map_from_threat(threat: dict) -> list:
    """Reconstruct a minimal zone list from blocked_zones for coordinate scanning."""
    # We don't have the full zone_map here; return empty so safe/avoid use blocked_zones only
    return []


def analyze_frame(terrain_stats: dict, zone_map: list) -> dict:
    """
    Full single-frame tactical analysis.
    Returns traversability, threat, recommendation, and UTC timestamp.
    """
    traversability = get_traversability_score(terrain_stats)
    threat = get_threat_assessment(terrain_stats, zone_map)

    # Pass zone_map into recommendation for safe/avoid zone detection
    trav_rating = traversability['rating']
    threat_level = threat['threat_level']

    if threat_level == 'HIGH' and trav_rating == 'LOW':
        primary_action = 'RETREAT'
    elif threat_level == 'HIGH' or trav_rating == 'LOW':
        primary_action = 'HOLD'
    elif threat_level == 'MEDIUM' or trav_rating == 'MEDIUM':
        primary_action = 'PROCEED WITH CAUTION'
    else:
        primary_action = 'ADVANCE'

    reasoning = (
        f"Traversability score is {traversability['score']:.1f}/100 ({trav_rating}), "
        f"indicating {traversability['recommendation'].lower()} "
        f"Threat assessment is {threat_level}: {threat['summary']}"
    )

    alert_coords = {(z['zone_row'], z['zone_col']) for z in threat['blocked_zones']}
    safe_zones = []
    avoid_zones = []
    for row_zones in zone_map:
        for zone in row_zones:
            coord = (zone['zone_row'], zone['zone_col'])
            ts = zone['traversability_score']
            in_alert = coord in alert_coords
            if ts > 60 and not in_alert:
                safe_zones.append({'row': zone['zone_row'], 'col': zone['zone_col'], 'score': ts})
            if ts < 30 or in_alert:
                avoid_zones.append({'row': zone['zone_row'], 'col': zone['zone_col'], 'score': ts})

    recommendation = {
        'primary_action': primary_action,
        'reasoning': reasoning,
        'safe_zones': safe_zones,
        'avoid_zones': avoid_zones,
    }

    return {
        'traversability': traversability,
        'threat': threat,
        'recommendation': recommendation,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    }


def analyze_video_summary(per_frame_stats: list) -> dict:
    """
    Aggregate tactical statistics across all processed video frames.
    """
    if not per_frame_stats:
        return {
            'avg_traversability_score': 0.0,
            'min_traversability_frame': None,
            'max_traversability_frame': None,
            'dominant_class': None,
            'alert_frequency_pct': 0.0,
            'terrain_change_rate': 0.0,
            'overall_recommendation': 'No data.',
        }

    frame_scores = []
    for stats in per_frame_stats:
        score = sum(
            CLASSES[c]['weight'] * stats['per_class'][c]['percentage']
            for c in stats['per_class']
            if c not in ABSENT_CLASS_IDS and CLASSES[c]['present_in_dataset']
        )
        frame_scores.append(score)

    avg_score = round(float(sum(frame_scores) / len(frame_scores)), 2)
    min_idx = int(frame_scores.index(min(frame_scores)))
    max_idx = int(frame_scores.index(max(frame_scores)))

    # Dominant class: highest total pixel percentage summed across all frames
    class_totals = {c: 0.0 for c in CLASSES if c not in ABSENT_CLASS_IDS}
    for stats in per_frame_stats:
        for c in class_totals:
            class_totals[c] += stats['per_class'][c]['percentage']
    dominant_cls_id = max(class_totals, key=class_totals.__getitem__)
    dominant_class = CLASSES[dominant_cls_id]['name']

    alert_frames = sum(1 for s in per_frame_stats if s['active_alerts'])
    alert_frequency_pct = round(100.0 * alert_frames / len(per_frame_stats), 2)

    # Terrain change rate: mean absolute difference in class percentages between consecutive frames
    change_rates = []
    for i in range(1, len(per_frame_stats)):
        diff = sum(
            abs(per_frame_stats[i]['per_class'][c]['percentage'] - per_frame_stats[i - 1]['per_class'][c]['percentage'])
            for c in CLASSES if c not in ABSENT_CLASS_IDS
        )
        change_rates.append(diff)
    terrain_change_rate = round(float(sum(change_rates) / len(change_rates)), 4) if change_rates else 0.0

    if avg_score >= 70:
        overall_recommendation = 'Zone generally passable. Advance recommended along high-score corridors.'
    elif avg_score >= 40:
        overall_recommendation = 'Mixed terrain. Use caution and prefer high-traversability zones.'
    else:
        overall_recommendation = 'Zone heavily obstructed across most frames. Hold or reroute.'

    return {
        'avg_traversability_score': avg_score,
        'min_traversability_frame': min_idx,
        'max_traversability_frame': max_idx,
        'dominant_class': dominant_class,
        'alert_frequency_pct': alert_frequency_pct,
        'terrain_change_rate': terrain_change_rate,
        'overall_recommendation': overall_recommendation,
    }
