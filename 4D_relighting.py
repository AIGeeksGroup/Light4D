from light4d.cli import main


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared after error")
        raise
