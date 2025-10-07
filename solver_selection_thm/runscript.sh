for i in {0..4}; do
    python solver_selection_thm/spe_runscript.py $i random
    python solver_selection_thm/spe_runscript.py $i solver_selection
    python solver_selection_thm/thm_runscript.py $i random
    python solver_selection_thm/thm_runscript.py $i solver_selection
done

for i in {0..4}; do
    python solver_selection_thm/spe_runscript.py $i expert
    python solver_selection_thm/thm_runscript.py $i expert
done
