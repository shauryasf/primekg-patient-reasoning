"""Neuro-symbolic progression modeling and patient reports.

The neural part is a survival GNN over the fused patient + PrimeKG graph. The
symbolic part is a set of survival rules learned over readable predicates, some
of which are PrimeKG facts (an altered gene that sits in a pathway). The two are
fused into one Cox model, and every patient report shows both the GNN-scored
knowledge-graph paths and the rules that fire for that patient.
"""
