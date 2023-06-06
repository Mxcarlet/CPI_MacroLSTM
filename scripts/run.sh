export CUDA_VISIBLE_DEVICES=0

cd ../

for model in MacroLSTM
do

for seq_len in 36 60 96
do

for preLen in 1
do

python -u main.py \
  --module $model \
  --seq_len $seq_len \
  --pred_len $preLen \
  --batch_size 32 \
  --epochs 100 \
  --driven_size 127
done
done

for seq_len in 60
do

#for preLen in 1
for preLen in 3 5 10
do

python -u main.py \
  --module $model \
  --seq_len $seq_len \
  --pred_len $preLen \
  --batch_size 32 \
  --epochs 100 \
  --driven_size 127
done
done

done

