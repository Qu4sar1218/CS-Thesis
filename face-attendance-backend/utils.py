from datetime import datetime

def format_student_name(first_name, middle_name, last_name):
    """Format student name as: Firstname Lastname M."""
    if not first_name or not last_name:
        return f"{first_name or ''} {last_name or ''}".strip()

    # Capitalize each word in first_name and last_name
    def capitalize(s):
        return ' '.join(word.capitalize() for word in s.split())

    capitalized_first = capitalize(first_name)
    capitalized_last = capitalize(last_name)

    last_parts = capitalized_last.split()
    if len(last_parts) > 1:
        # Handle multiple words in last name
        middle_initial = f" {middle_name[0].upper()}." if middle_name else ""
        return f"{capitalized_first} {last_parts[0]}{middle_initial} {' '.join(last_parts[1:])}".strip()
    else:
        # Standard format
        middle_initial = f" {middle_name[0].upper()}." if middle_name else ""
        return f"{capitalized_first}{middle_initial} {capitalized_last}".strip()

def is_class_scheduled_today(schedule: str) -> bool:
    """
    Check if a class is scheduled for today based on its schedule string.

    Args:
        schedule: Schedule string like "MWF 9:00-10:00"

    Returns:
        bool: True if scheduled for today, False otherwise
    """
    if not schedule:
        return False

    # Split schedule into days and time parts
    parts = schedule.split()
    if not parts:
        return False

    days_str = parts[0]  # e.g., "MWF" or "ThF"

    # Day codes mapping (weekday() returns 0=Monday, 1=Tuesday, etc.)
    day_codes = ['M', 'T', 'W', 'Th', 'F', 'S', 'Su']
    today_weekday = datetime.now().weekday()
    today_code = day_codes[today_weekday]

    # Parse days_str into individual day codes
    possible_days = ['Su', 'Th', 'M', 'T', 'W', 'F', 'S']  # Check longer codes first
    scheduled_days = []
    i = 0
    while i < len(days_str):
        found = False
        for day in possible_days:
            if days_str.startswith(day, i):
                scheduled_days.append(day)
                i += len(day)
                found = True
                break
        if not found:
            i += 1  # Skip invalid character

    # Check if today_code is in scheduled_days
    return today_code in scheduled_days
