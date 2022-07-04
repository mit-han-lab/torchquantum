.. role:: hidden
    :class: hidden-section
.. currentmodule:: {{ module }}

{{ name | underline }}

.. autoclass:: {{ name }}
    :show-inheritance:

    .. automethod:: __init__

{% block methods_summary %}
{% set wanted_methods = (methods | reject('in', inherited_members) | reject('==', '__init__') | list) %}
{% if wanted_methods %}
    .. rubric:: Methods

{% for item in wanted_methods %}
    .. automethod:: {{ name }}.{{ item }}

{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_summary %}
    .. rubric:: Attributes

{% set wanted_attributes = (attributes | reject('in', inherited_members) | list) %}
{% if wanted_attributes %}
{% for item in wanted_attributes %}
    .. autoattribute:: {{ name }}.{{ item }} 

{%- endfor %}
{% endif %}
{% endblock -%}
