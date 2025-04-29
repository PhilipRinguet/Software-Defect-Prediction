"""Unified plotting interface with configuration handling."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import load_config

class PlotManager:
    def __init__(self):
        config = load_config()
        self.viz_config = config['visualization']
        
        # Set global plotting style
        sns.set_style(self.viz_config['style'])  # Use seaborn's style setting
        sns.set_palette(self.viz_config['colors']['primary_palette'])
    
    def get_figure_size(self, plot_type='default'):
        """Get figure size from config for specific plot type."""
        return self.viz_config['figure_sizes'].get(plot_type, 
                                                 self.viz_config['figure_sizes']['default'])
    
    def setup_figure(self, plot_type='default'):
        """Create and configure a new figure."""
        fig = plt.figure(figsize=self.get_figure_size(plot_type))
        return fig
    
    def save_figure(self, fig, name, subdir=None):
        """Save figure in all configured formats."""
        base_dir = self.viz_config['output_dir']
        if subdir:
            save_dir = os.path.join(base_dir, subdir)
        else:
            save_dir = base_dir
            
        os.makedirs(save_dir, exist_ok=True)
        
        for fmt in self.viz_config['formats']:
            path = os.path.join(save_dir, f'{name}.{fmt}')
            fig.savefig(path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        
        plt.close(fig)
    
    def save_plots(self, plots_dict, subdir=None):
        """Save multiple plots from a dictionary."""
        for name, fig in plots_dict.items():
            self.save_figure(fig, name, subdir)

# Global plot manager instance
plot_manager = PlotManager()