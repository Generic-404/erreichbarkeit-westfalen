class AccessibilityError(Exception):
    """Basisklasse für Konfigurations-, Daten- und Abhängigkeitsfehler."""


class ConfigurationError(AccessibilityError):
    """Die YAML-Konfiguration ist ungültig oder unvollständig."""


class DataValidationError(AccessibilityError):
    """Eine Eingabedatei ist fachlich nicht für die Analyse verwendbar."""


class DependencyError(AccessibilityError):
    """Eine optionale Laufzeitabhängigkeit oder ein externes Werkzeug fehlt."""
