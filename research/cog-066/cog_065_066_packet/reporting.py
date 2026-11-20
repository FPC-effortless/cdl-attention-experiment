"""Deterministic, full-information receipt reporting ceiling for the 066 run."""

STATUS=('completed','blocked','unknown')

def template(fields):
    v=dict(fields)
    return f"parcel p{v['request']} is {STATUS[v['receipt']]} at version {v['version']} with {v['progress']} of {v['total']} steps done"

