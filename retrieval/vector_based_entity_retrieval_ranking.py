#!/usr/bin/env python
import argparse
import sys
import asyncio
import os

def main():
    parser = argparse.ArgumentParser(description="KG-RAG4SM Main Entry Point")
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    # Sub-command for similarity search
    parser_sim = subparsers.add_parser('similarity_search', help='Run similarity search')
    parser_sim.add_argument("--test_mode", action="store_true", help="Enable test mode for similarity search")
    parser_sim.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")

    # Sub-command for BFS paths
    parser_bfs = subparsers.add_parser('bfs_paths', help='Find BFS paths between entities')
    parser_bfs.add_argument("--max_hops", type=int, default=3, help="Maximum number of hops for BFS paths")
    parser_bfs.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")

    # Sub-command for path ranking
    parser_rank = subparsers.add_parser('path_ranking', help='Rank BFS paths and export results')
    parser_rank.add_argument("--bfs_results", type=str, default="cms_wikidata_paths_final_full.json", help="Input BFS results JSON file")
    parser_rank.add_argument("--question_similar_data", type=str, default="cms_wikidata_similar_full.json", help="Input question similarity data JSON file")
    parser_rank.add_argument("--output_csv", type=str, default="pruned_bfs_results.csv", help="Output CSV filename")
    parser_rank.add_argument("--output_json", type=str, default="pruned_bfs_results.json", help="Output JSON filename")
    parser_rank.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")

    args = parser.parse_args()

    try:
        if args.command == 'similarity_search':
            print(f"Running similarity search{' in test mode' if args.test_mode else ''}")
            from modules import similarity_search
            cmd_args = [sys.argv[0]]
            if args.test_mode:
                cmd_args.append("--test_mode")
            if args.limit:
                cmd_args.extend(["--limit", str(args.limit)])
            sys.argv = cmd_args
            similarity_search.main()
            
        elif args.command == 'bfs_paths':
            print(f"Running BFS paths with max_hops={args.max_hops}")
            from modules import bfs_paths
            cmd_args = [sys.argv[0], "--max_hops", str(args.max_hops)]
            if args.limit:
                cmd_args.extend(["--limit", str(args.limit)])
            sys.argv = cmd_args
            asyncio.run(bfs_paths.main())
            
        elif args.command == 'path_ranking':
            print(f"Running path ranking")
            # Check if input files exist
            for filepath in [args.bfs_results, args.question_similar_data]:
                if not os.path.exists(filepath):
                    print(f"Error: Input file not found: {filepath}")
                    return
                    
            from modules import path_ranking
            cmd_args = [
                sys.argv[0],
                "--bfs_results", args.bfs_results,
                "--question_similar_data", args.question_similar_data,
                "--output_csv", args.output_csv,
                "--output_json", args.output_json
            ]
            if args.limit:
                cmd_args.extend(["--limit", str(args.limit)])
            sys.argv = cmd_args
            path_ranking.main()
            
        else:
            parser.print_help()
            
    except ImportError as e:
        print(f"Error: Could not import required module: {e}")
        print("Please make sure you're in the correct directory and all dependencies are installed.")
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()