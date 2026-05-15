COMPRESSION_STRATEGIES = {
    "powersgd": {
        "description": "Low-rank approximation of gradient matrices",
        "bandwidth_reduction": "40-60%",
        "memory_overhead": "High (requires >6GB VRAM)",
        "paper": "PowerSGD: Practical Low-Rank Gradient Compression (Vogels et al., 2019)",
    },
    "topk": {
        "description": "Transmit only top-K largest gradients",
        "bandwidth_reduction": "up to 99%",
        "memory_overhead": "Low",
        "paper": "Deep Gradient Compression (Lin et al., 2018)",
    },
}