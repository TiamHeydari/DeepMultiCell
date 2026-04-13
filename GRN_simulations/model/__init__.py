from .deepsets_vae import DeepSet_Linear_Layer, DeepSet_VAE
from .models import DeepSet_Linear_Layer, DeepSet_Auto_encoder_v1, DeepSet_Auto_encoder_v2_SetTransformer,DeepSet_Auto_encoder_v3_SetTransformer, DeepSet_Auto_encoder_v4_SetTransformer
from .data_loading import (
	ColonyFromAnnDataDataset,
	build_dataloaders,
	create_batch,
	downsample_adata_by_colony,
	load_adata,
	prepare_pointcloud_data,
)
# from .visualizations import (
# 	collect_all_sets_z,
# 	pca_and_highlight_classes,
# 	show_reconstruction_examples,
# 	umap_and_highlight_classes,
# 	visualize_samples,
# 	visualize_samples_3d,
# )


from model.visualizations import (
    collect_all_sets_z,
    pca_and_highlight_classes,
    pca_plot_2d,
    pca_plot_3d,
    show_reconstruction_examples,
    umap_and_highlight_classes,
    visualize_samples,
    visualize_samples_3d,
)

from .training import (
	eval_epoch,
	fit_model,
	get_beta_linear,
	make_optimizer_scheduler,
	plot_training_history,
	sinkhorn_loss,
	test,
	train,
	train_epoch,
)
