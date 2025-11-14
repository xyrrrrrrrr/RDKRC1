# python Learn_MCDKN_gu.py --K_train_samples 1000
# # python Learn_MCDKN_gu.py --K_train_samples 5000
# # python Learn_MCDKN_gu.py --K_train_samples 20000

# # python Learn_Koopman_with_KlinearEig.py --K_train_samples 200
# # python Learn_Koopman_with_KlinearEig.py --K_train_samples 1000
# # python Learn_Koopman_with_KlinearEig.py --K_train_samples 5000
# python Learn_Koopman_with_KlinearEig.py --K_train_samples 20000 --env "CartPole-v1"
# python Learn_Koopman_with_KlinearEig.py --K_train_samples 50000

# python Learn_Knonlinear.py --K_train_samples 200
# python Learn_Knonlinear.py --K_train_samples 1000
# # python Learn_Knonlinear.py --K_train_samples 5000
# python Learn_Knonlinear.py --K_train_samples 20000 --env "CartPole-v1"
# # python Learn_Knonlinear.py --K_train_samples 50000

# # python Learn_KoopmanNonlinear_with_KlinearEig.py --K_train_samples 200
# # python Learn_KoopmanNonlinear_with_KlinearEig.py --K_train_samples 1000
# # python Learn_KoopmanNonlinear_with_KlinearEig.py --K_train_samples 5000
# python Learn_KoopmanNonlinear_with_KlinearEig.py --K_train_samples 20000 --env "CartPole-v1"
# # python Learn_KoopmanNonlinear_with_KlinearEig.py --K_train_samples 50000

# # python Learn_KoopmanNonlinearA_with_KlinearEig.py --K_train_samples 200
# # python Learn_KoopmanNonlinearA_with_KlinearEig.py --K_train_samples 1000
# # python Learn_KoopmanNonlinearA_with_KlinearEig.py --K_train_samples 5000
# python Learn_KoopmanNonlinearA_with_KlinearEig.py --K_train_samples 20000 --env "CartPole-v1"
# # python Learn_KoopmanNonlinearA_with_KlinearEig.py --K_train_samples 50000

# # python Learn_Knonlinear_RNN.py --K_train_samples 200
# # python Learn_Knonlinear_RNN.py --K_train_samples 1000
# # python Learn_Knonlinear_RNN.py --K_train_samples 5000
# python Learn_Knonlinear_RNN.py --K_train_samples 20000 --env "CartPole-v1"
# # python Learn_Knonlinear_RNN.py --K_train_samples 50000

# # python Learn_DKN_gxu.py --K_train_samples 200
# # python Learn_DKN_gxu.py --K_train_samples 1000
# # python Learn_DKN_gxu.py --K_train_samples 5000
# python Learn_DKN_gxu.py --K_train_samples 20000 --env "CartPole-v1"
# python Learn_DKN_gxu.py --K_train_samples 50000

python Learn_DKN_gxu.py --K_train_samples 20000  --lambda_geom 0.0 --lambda_control 0.1 --lambda_recon 0.05
python Learn_DKN_gxu.py --K_train_samples 20000  --lambda_geom 0.1 --lambda_control 0.1 --lambda_recon 0.05
python Learn_DKN_gxu.py --K_train_samples 20000  --lambda_geom 0.2 --lambda_control 0.1 --lambda_recon 0.05