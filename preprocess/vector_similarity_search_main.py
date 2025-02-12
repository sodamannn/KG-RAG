#!/usr/bin/env python
import argparse
import sys
import asyncio

def main():
    parser = argparse.ArgumentParser(description="KG-RAG4SM Main Entry Point")
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    # Sub-command for similarity search
    parser_sim = subparsers.add_parser('similarity_search', help='Run similarity search')
    parser_sim.add_argument("--test_mode", action="store_true", help="Enable test mode for similarity search")

    # Sub-command for BFS paths
    parser_bfs = subparsers.add_parser('bfs_paths', help='Find BFS paths between entities')
    parser_bfs.add_argument("--max_hops", type=int, default=3, help="Maximum number of hops for BFS paths")

    # Sub-command for path ranking
    parser_rank = subparsers.add_parser('path_ranking', help='Rank BFS paths and export results')
    parser_rank.add_argument("--bfs_results", type=str, default="cms_wikidata_paths_final_full.json", help="Input BFS results JSON file")
    parser_rank.add_argument("--question_similar_data", type=str, default="cms_wikidata_similar_full.json", help="Input question similarity data JSON file")
    parser_rank.add_argument("--output_csv", type=str, default="pruned_bfs_results.csv", help="Output CSV filename")
    parser_rank.add_argument("--output_json", type=str, default="pruned_bfs_results.json", help="Output JSON filename")

    args = parser.parse_args()

    if args.command == 'similarity_search':
        from modules import similarity_search
        if args.test_mode:
            sys.argv = [sys.argv[0], "--test_mode"]
        else:
            sys.argv = [sys.argv[0]]
        similarity_search.main()
    elif args.command == 'bfs_paths':
        from modules import bfs_paths
        sys.argv = [sys.argv[0], "--max_hops", str(args.max_hops)]
        asyncio.run(bfs_paths.main())
    elif args.command == 'path_ranking':
        from modules import path_ranking
        sys.argv = [sys.argv[0],
                    "--bfs_results", args.bfs_results,
                    "--question_similar_data", args.question_similar_data,
                    "--output_csv", args.output_csv,
                    "--output_json", args.output_json]
        path_ranking.main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
