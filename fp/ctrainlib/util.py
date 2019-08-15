from copy import copy
from typing import List, Dict, Optional, Any, Callable, Union

PICKLE_PROTOCOL = 4


class Type:
    """
    Class to define a type via a Callable with a default value.

    Parameters
    ----------
    type_func : Callable
        The function used to "cast" the value
    default : Any
        An optional default value for this type
    """
    __slots__ = ('type', 'default')

    def __init__(self, type_func: Callable, default: Any):
        self.type = type_func
        self.default = default

    def process(self, value: str) -> Optional[Any]:
        """
        Casts the given value with the Callable in self.type.

        Parameters
        ----------
        value : str
            The value to cast

        Returns
        -------
        Optional[Any]
            The casted value or None, if casting was not possible
        """
        try:
            return self.type(value)
        except ValueError:
            pass


class Choices:
    """
    Class to define a list of possible string choices and a default value.

    Parameters
    ----------
    choices : List[str]
        A list of string representing the possible choices
    default : str
        An optional default string value
    """
    __slots__ = ('choices', 'default')

    def __init__(self, choices: List[str], default: str):
        self.choices = choices
        self.default = default

    def process(self, value: str) -> Optional[str]:
        """
        Checks if value is in self.choices.

        Parameters
        ----------
        value : str
            The value to check

        Returns
        -------
        Optional[str]
            The value itself, if it is contained in self.choices, None otherwise
        """
        if value in self.choices:
            return value


def boolean_type(value: str) -> Optional[bool]:
    """
    Converts a string value into a boolean value, if possible.

    The string is converted into True, if its lowercase value equals '1', 'true' or 't'.
    It is converted into False, if its lowercase value equals '0', 'false' or 'f'.
    If value equals non of these values, None is returned.

    Parameters
    ----------
    value : str
        The string value to be casted into a boolean value

    Returns
    -------
    Optional[bool]
        True or False regarding the rules mentioned above, None otherwise
    """

    if value.lower() in ('1', 'true', 't'):
        return True
    if value.lower() in ('0', 'false', 'f'):
        return False


def int_or_float_type(value: str) -> Optional[Union[int, float]]:
    """
    Converts a string value into an integer or a float value between 0.0 and 1.0 (inclusive).

    Parameters
    ----------
    value : str
        The string value to be converted

    Returns
    -------
    Optional[Union[int, float]]
        If value represents an integer value, an integer is returned. Otherwise a float value is returned,
        if value is convertible into float and the converted float value is between 0.0 and 1.0.
        Else returning None.

    """

    if value.isdigit():
        return int(value)
    try:
        v = float(value)
    except ValueError:
        pass
    else:
        if 0.0 <= v <= 1.0:
            return v


def max_features_type(value: str) -> Optional[Union[str, int, float]]:
    """
    Special function to parse a string value into the possible types of the max_features parameter from scikit-learn.

    If the string equals 'auto', 'sqrt' or 'log2', the value is returned unchanged.
    If not, the function ``int_or_float_type(value)`` is called and the result is returned.

    Parameters
    ----------
    value : str
        The string value to be parsed

    Returns
    -------
    Optional[Union[str, int, float]]
        A string regarding the rules mentioned above, otherwise an integer, float or None is returned
        (see function ``int_or_float_type()``).

    See Also
    --------
    ctrainlib.classifiers.int_or_float_type : Converts a string value into an integer
                                              or a float value between 0.0 and 1.0 (inclusive).
    """

    if value in ['auto', 'sqrt', 'log2']:
        return value
    return int_or_float_type(value)


def int_list_type(value: str) -> Optional[List[int]]:
    """
    Converts a string with comma separated int values into a list of integers.

    Parameters
    ----------
    value : str
        The string value to be converted

    Returns
    -------
    Optional[List[int]]
        If the string value is a comma separated list of integers, a list of integers is returned.
        Else returning None.
    """

    try:
        arr = [int(x) for x in value.split(',')]
        return arr
    except ValueError:
        pass


def float_list_type(value: str) -> Optional[List[float]]:
    """
    Converts a string with comma separated float values into a list of floats.

    Parameters
    ----------
    value : str
        The string value to be converted

    Returns
    -------
    Optional[List[float]]
        If the string value is a comma separated list of floats, a list of floats is returned.
        Else returning None.
    """

    try:
        arr = [float(x) for x in value.split(',')]
        return arr
    except ValueError:
        pass


def parse_options(cls_info: Any, options: List[str]) -> Union[Dict[str, Any], str]:
    """
    Parsing options for the given classifier or regressor class.

    Options have to be in the format `KEY=VALUE`. For all values not given in `options`,
    the default values of `cls_info` are used.

    Parameters
    ----------
    cls_info : Any
        Classifier/regressor class from ctrainlib.classifiers/ctrainlib.regressors,
        containing the possible parameters with default values
    options : List[str]
        List of strings containing options in format `KEY=VALUE`

    Returns
    -------
    Union[Dict[str, Any], str]
        A dictionary containing all (chosen and default) options for the given class,
        if no errors occur, a string containing an error message otherwise.
    """

    opt_dict = {}
    for option in options:
        spl = option.split('=')
        if len(spl) != 2:
            return f'Wrong option format: {option}'
        key, value = spl
        arg_type = cls_info.params.get(key)
        if not arg_type:
            return f'Option not available for chosen classifier/regressor: {option}'
        value = arg_type.process(value)
        if not value:
            return f'Wrong type or value for option: {option}'
        opt_dict[key] = value
    for option in cls_info.params:
        if option not in opt_dict:
            opt_dict[option] = copy(cls_info.params[option].default)
    return opt_dict
