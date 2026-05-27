from datetime import datetime

from common import settings


def now_capture_id() -> str:
    return "capture_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def now_session_id() -> str:
    return "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def now_event_id(prefix: str = "event") -> str:
    return prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def system_path() -> str:
    return "system"


def company_path(company_id: str = settings.COMPANY_ID) -> str:
    return f"companies/{company_id}"


def site_path(site_id: str = settings.SITE_ID) -> str:
    return f"sites/{site_id}"


def robot_path(robot_id: str = settings.ROBOT_ID) -> str:
    return f"robots/{robot_id}"


def twin_static_path(site_id: str = settings.SITE_ID) -> str:
    return f"twin_static/{site_id}"


def twin_state_path(site_id: str = settings.SITE_ID) -> str:
    return f"twin_state/{site_id}"


def current_inspection_path(site_id: str = settings.SITE_ID) -> str:
    return f"twin_state/{site_id}/current_inspection"


def inspections_path(site_id: str = settings.SITE_ID) -> str:
    return f"inspections/{site_id}/captures"


def inspection_capture_path(capture_id: str, site_id: str = settings.SITE_ID) -> str:
    return f"inspections/{site_id}/captures/{capture_id}"


def inspection_markers_path(capture_id: str, site_id: str = settings.SITE_ID) -> str:
    return f"inspections/{site_id}/captures/{capture_id}/markers"




def sessions_path(site_id: str = settings.SITE_ID) -> str:
    return f"inspections/{site_id}/sessions"


def inspection_session_path(session_id: str, site_id: str = settings.SITE_ID) -> str:
    return f"inspections/{site_id}/sessions/{session_id}"


def session_capture_path(
    session_id: str,
    workstation_id: str,
    capture_id: str,
    site_id: str = settings.SITE_ID,
) -> str:
    return (
        f"inspections/{site_id}/sessions/{session_id}/"
        f"workstations/{workstation_id}/captures/{capture_id}"
    )


def session_markers_path(
    session_id: str,
    workstation_id: str,
    capture_id: str,
    site_id: str = settings.SITE_ID,
) -> str:
    return (
        f"inspections/{site_id}/sessions/{session_id}/"
        f"workstations/{workstation_id}/captures/{capture_id}/markers"
    )


def robot_command_current_path(robot_id: str = settings.ROBOT_ID) -> str:
    return f"robot_commands/{robot_id}/current"


def robot_command_history_path(robot_id: str = settings.ROBOT_ID) -> str:
    return f"robot_commands/{robot_id}/history"


def events_path(site_id: str = settings.SITE_ID) -> str:
    return f"events/{site_id}"




def indexes_path(site_id: str = settings.SITE_ID) -> str:
    return f"indexes/{site_id}"


def index_latest_path(site_id: str = settings.SITE_ID) -> str:
    return f"indexes/{site_id}/latest"


def index_capture_lookup_path(
    capture_id: str,
    site_id: str = settings.SITE_ID,
) -> str:
    return f"indexes/{site_id}/capture_lookup/{capture_id}"


def index_captures_by_date_path(
    date_key: str,
    capture_id: str,
    site_id: str = settings.SITE_ID,
) -> str:
    return f"indexes/{site_id}/captures_by_date/{date_key}/{capture_id}"


def index_captures_by_workstation_path(
    workstation_id: str,
    capture_id: str,
    site_id: str = settings.SITE_ID,
) -> str:
    return f"indexes/{site_id}/captures_by_workstation/{workstation_id}/{capture_id}"


def index_unresolved_defect_path(
    defect_key: str,
    site_id: str = settings.SITE_ID,
) -> str:
    return f"indexes/{site_id}/defects_by_status/unresolved/{defect_key}"


def external_exports_path(company_id: str = settings.COMPANY_ID) -> str:
    return f"external_exports/{company_id}"


def legacy_linestatus_path() -> str:
    return "legacy/linestatus"


def legacy_linestatus_workstation_path(workstation_name: str) -> str:
    return f"legacy/linestatus/{workstation_name}"


def storage_inspection_base_path(
    capture_id: str,
    session_id: str = "",
    workstation_id: str = "",
) -> str:
    if session_id and workstation_id:
        return (
            f"companies/{settings.COMPANY_ID}/"
            f"sites/{settings.SITE_ID}/"
            f"inspections/sessions/{session_id}/"
            f"workstations/{workstation_id}/"
            f"captures/{capture_id}"
        )

    return (
        f"companies/{settings.COMPANY_ID}/"
        f"sites/{settings.SITE_ID}/"
        f"inspections/{capture_id}"
    )


def storage_background_js_path(
    capture_id: str,
    session_id: str = "",
    workstation_id: str = "",
) -> str:
    return f"{storage_inspection_base_path(capture_id, session_id, workstation_id)}/background/bg.js"


def storage_rgb_image_path(
    capture_id: str,
    session_id: str = "",
    workstation_id: str = "",
) -> str:
    return f"{storage_inspection_base_path(capture_id, session_id, workstation_id)}/images/rgb.jpg"


def storage_annotated_image_path(
    capture_id: str,
    session_id: str = "",
    workstation_id: str = "",
) -> str:
    return f"{storage_inspection_base_path(capture_id, session_id, workstation_id)}/images/annotated.jpg"


def storage_pointcloud_path(
    capture_id: str,
    session_id: str = "",
    workstation_id: str = "",
) -> str:
    return f"{storage_inspection_base_path(capture_id, session_id, workstation_id)}/pointcloud/pointcloud.pcd"


def storage_report_path(
    capture_id: str,
    session_id: str = "",
    workstation_id: str = "",
) -> str:
    return f"{storage_inspection_base_path(capture_id, session_id, workstation_id)}/report/result.json"
