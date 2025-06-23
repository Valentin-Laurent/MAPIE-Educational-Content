for dimension in tutorial use-case
do
    sed -n '/correction/!p' notebooks/regression-$dimension-correction.ipynb > notebooks/regression-$dimension.ipynb
done
