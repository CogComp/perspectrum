"""
template filters for raw json claim data
https://docs.djangoproject.com/en/dev/howto/custom-template-tags/#writing-custom-template-filters
"""

from django import template

register = template.Library()


@register.filter
def dict_get_item(dict, key):
    return dict[key]

@register.filter
def get_pro_description(perspective):
    if "argument" in perspective:
        arg = perspective["argument"]
        if "description" in arg:
            return arg["description"]
        else:
            return []
    else:
        return []

@register.filter
def get_con_description(perspective):
    if "argument" in perspective:
        arg = perspective["counter_argument"]
        if "description" in arg:
            return arg["description"]
        else:
            return []
    else:
        return []