"""
H3 Grid Manager for Lightning Strike Prediction

This module provides utilities for working with H3 hexagonal grids:
- Converting lat/lon to H3 cells
- Finding neighbor cells at different ring levels
- Calculating distances between cells
- Managing cell properties and metadata

Designed to work with DataFrames for offline training, but structured
to be adaptable for Redis-based streaming later.
"""

import h3
import numpy as np
import pandas as pd
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class CellInfo:
    """Information about an H3 cell."""
    cell_id: str
    lat: float
    lon: float
    resolution: int
    area_km2: float
    
    def __hash__(self):
        return hash(self.cell_id)


class GridManager:
    """
    Manages H3 hexagonal grid operations.
    
    Key Operations:
    1. Convert lat/lon to H3 cells
    2. Find neighbors at various ring levels
    3. Calculate distances (grid and geographic)
    4. Cache frequently accessed data
    
    Usage:
        grid = GridManager(resolution=7)
        cell_id = grid.lat_lon_to_cell(28.5, -81.5)
        neighbors = grid.get_neighbors(cell_id, ring=1)
    """
    
    def __init__(self, resolution: int = 7):
        """
        Initialize grid manager with specified H3 resolution.
        
        Args:
            resolution: H3 resolution level (0-15). Default 7 (~5.16 km²)
        """
        self.resolution = resolution
        # Cache for neighbor lookups (expensive operation)
        self._neighbor_cache: Dict[Tuple[str, int], Set[str]] = {}
        
    def lat_lon_to_cell(self, lat: float, lon: float) -> str:
        """
        Convert latitude/longitude to H3 cell ID.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            H3 cell ID as hex string
        """
        return h3.latlng_to_cell(lat, lon, self.resolution)
    
    def cell_to_lat_lon(self, cell_id: str) -> Tuple[float, float]:
        """
        Convert H3 cell ID to center lat/lon.
        
        Args:
            cell_id: H3 cell ID
            
        Returns:
            Tuple of (latitude, longitude)
        """
        return h3.cell_to_latlng(cell_id)
    
    def add_cell_column(self, df: pd.DataFrame, 
                       lat_col: str = 'lat', 
                       lon_col: str = 'lon',
                       cell_col: str = 'h3_cell') -> pd.DataFrame:
        """
        Add H3 cell ID column to DataFrame.
        
        This is the primary way to convert strike data to grid cells.
        
        Args:
            df: DataFrame with lat/lon columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            cell_col: Name for new cell ID column
            
        Returns:
            DataFrame with added cell_col
            
        Example:
            df = grid.add_cell_column(strikes_df)
            # Now df has 'h3_cell' column
        """
        df = df.copy()
        df[cell_col] = df.apply(
            lambda row: self.lat_lon_to_cell(row[lat_col], row[lon_col]), 
            axis=1
        )
        return df
    
    def get_neighbors(self, cell_id: str, ring: int = 1) -> Set[str]:
        """
        Get neighbor cells at specified ring distance.
        
        Ring levels:
        - ring=1: 6 immediate neighbors
        - ring=2: 12 additional neighbors (18 total)
        - ring=k: 6*k neighbors in that ring
        
        Results are cached for performance.
        
        Args:
            cell_id: Center H3 cell ID
            ring: Ring level (1, 2, 3...)
            
        Returns:
            Set of neighbor cell IDs (excludes center cell)
        """
        cache_key = (cell_id, ring)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]
        
        # Get all cells up to this ring level (returns list, convert to set)
        neighbors = set(h3.grid_disk(cell_id, ring))
        # Remove the center cell itself
        neighbors.discard(cell_id)
        
        self._neighbor_cache[cache_key] = neighbors
        return neighbors
    
    def get_neighbors_up_to_ring(self, cell_id: str, max_ring: int = 2) -> Set[str]:
        """
        Get all neighbors up to and including specified ring.
        
        Args:
            cell_id: Center H3 cell ID
            max_ring: Maximum ring level
            
        Returns:
            Set of all neighbor cell IDs up to max_ring (excludes center)
            
        Example:
            # Get all cells within 2 rings (6 + 12 = 18 neighbors)
            neighbors = grid.get_neighbors_up_to_ring(cell_id, max_ring=2)
        """
        all_neighbors = set()
        for ring in range(1, max_ring + 1):
            all_neighbors.update(self.get_neighbors(cell_id, ring))
        return all_neighbors
    
    def get_neighbor_counts(self, df: pd.DataFrame, 
                           cell_col: str = 'h3_cell',
                           rings: List[int] = [1, 2]) -> pd.DataFrame:
        """
        For each cell, count strikes in neighboring cells.
        
        This is used for spatial features. Returns a DataFrame with
        neighbor counts for each ring level.
        
        Args:
            df: DataFrame with cell_col and strike counts
            cell_col: Column name with H3 cell IDs
            rings: List of ring levels to calculate
            
        Returns:
            DataFrame with columns: cell_id, neighbor_count_ring1, neighbor_count_ring2, etc.
            
        Example:
            # Get strike counts in neighboring cells
            neighbor_counts = grid.get_neighbor_counts(strikes_df)
        """
        # Count strikes per cell
        cell_counts = df[cell_col].value_counts().to_dict()
        
        # Get unique cells
        unique_cells = df[cell_col].unique()
        
        results = []
        for cell in unique_cells:
            row = {'cell_id': cell}
            
            for ring in rings:
                neighbors = self.get_neighbors(cell, ring)
                # Sum strikes in all neighbors at this ring
                neighbor_count = sum(cell_counts.get(n, 0) for n in neighbors)
                row[f'neighbor_count_ring{ring}'] = neighbor_count
                
                # Also calculate average per neighbor
                row[f'neighbor_density_ring{ring}'] = neighbor_count / len(neighbors) if neighbors else 0
                
                # Max strikes in any single neighbor
                neighbor_counts_list = [cell_counts.get(n, 0) for n in neighbors]
                row[f'max_neighbor_ring{ring}'] = max(neighbor_counts_list) if neighbor_counts_list else 0
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def calculate_grid_distance(self, cell1: str, cell2: str) -> int:
        """
        Calculate grid distance between two cells (number of steps).
        
        Args:
            cell1: First H3 cell ID
            cell2: Second H3 cell ID
            
        Returns:
            Integer grid distance (minimum steps to get from cell1 to cell2)
        """
        return h3.grid_distance(cell1, cell2)
    
    def calculate_geographic_distance_km(self, cell1: str, cell2: str) -> float:
        """
        Calculate geographic distance between two cells in kilometers.
        
        Uses haversine formula on cell centers.
        
        Args:
            cell1: First H3 cell ID
            cell2: Second H3 cell ID
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = self.cell_to_lat_lon(cell1)
        lat2, lon2 = self.cell_to_lat_lon(cell2)
        return self._haversine_distance(lat1, lon1, lat2, lon2)
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two lat/lon points.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_cell_info(self, cell_id: str) -> CellInfo:
        """
        Get detailed information about a cell.
        
        Args:
            cell_id: H3 cell ID
            
        Returns:
            CellInfo dataclass with cell properties
        """
        lat, lon = self.cell_to_lat_lon(cell_id)
        area_m2 = h3.cell_area(cell_id, unit='m^2')
        area_km2 = area_m2 / 1_000_000
        
        return CellInfo(
            cell_id=cell_id,
            lat=lat,
            lon=lon,
            resolution=self.resolution,
            area_km2=area_km2
        )
    
    def get_cell_boundary(self, cell_id: str) -> List[Tuple[float, float]]:
        """
        Get the boundary vertices of a cell (for visualization).
        
        Args:
            cell_id: H3 cell ID
            
        Returns:
            List of (lat, lon) tuples forming the hexagon boundary
        """
        return h3.cell_to_boundary(cell_id)
    
    def find_cells_within_radius(self, center_lat: float, center_lon: float,
                                 radius_km: float) -> Set[str]:
        """
        Find all cells within a radius of a point.
        
        Useful for defining prediction areas or analyzing local patterns.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            Set of cell IDs within the radius
        """
        # Start with center cell
        center_cell = self.lat_lon_to_cell(center_lat, center_lon)
        
        # Estimate how many rings we need
        # H3 resolution 7 has ~2.7km edge length, so radius/2.7 is rough estimate
        estimated_rings = int(radius_km / 2.7) + 1
        
        # Get all cells in that many rings
        candidate_cells = h3.grid_disk(center_cell, estimated_rings)
        
        # Filter by actual distance
        cells_within_radius = set()
        for cell in candidate_cells:
            lat, lon = self.cell_to_lat_lon(cell)
            distance = self._haversine_distance(center_lat, center_lon, lat, lon)
            if distance <= radius_km:
                cells_within_radius.add(cell)
        
        return cells_within_radius
    
    def clear_cache(self):
        """Clear the neighbor cache (useful if memory constrained)."""
        self._neighbor_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the neighbor cache."""
        return {
            'cached_lookups': len(self._neighbor_cache),
            'unique_cells': len(set(k[0] for k in self._neighbor_cache.keys())),
            'cache_size_bytes': sum(len(v) * 8 for v in self._neighbor_cache.values())
        }


# Convenience functions for quick operations
def lat_lon_to_cell(lat: float, lon: float, resolution: int = 7) -> str:
    """Quick conversion from lat/lon to H3 cell."""
    return h3.latlng_to_cell(lat, lon, resolution)


def cell_to_lat_lon(cell_id: str) -> Tuple[float, float]:
    """Quick conversion from H3 cell to lat/lon."""
    return h3.cell_to_latlng(cell_id)


if __name__ == "__main__":
    # Test the grid manager
    print("=" * 80)
    print("GRID MANAGER TEST")
    print("=" * 80)
    
    grid = GridManager(resolution=7)
    
    # Test lat/lon conversion
    test_lat, test_lon = 28.5, -81.5  # Orlando, FL area
    cell = grid.lat_lon_to_cell(test_lat, test_lon)
    print(f"\nTest location: ({test_lat}, {test_lon})")
    print(f"H3 Cell: {cell}")
    
    # Get cell info
    info = grid.get_cell_info(cell)
    print(f"\nCell Info:")
    print(f"  Center: ({info.lat:.4f}, {info.lon:.4f})")
    print(f"  Area: {info.area_km2:.3f} km²")
    
    # Test neighbors
    ring1_neighbors = grid.get_neighbors(cell, ring=1)
    ring2_neighbors = grid.get_neighbors(cell, ring=2)
    print(f"\nNeighbors:")
    print(f"  Ring 1: {len(ring1_neighbors)} cells")
    print(f"  Ring 2: {len(ring2_neighbors)} cells")
    print(f"  Total up to ring 2: {len(grid.get_neighbors_up_to_ring(cell, 2))} cells")
    
    # Test distance
    if ring1_neighbors:
        neighbor = list(ring1_neighbors)[0]
        grid_dist = grid.calculate_grid_distance(cell, neighbor)
        geo_dist = grid.calculate_geographic_distance_km(cell, neighbor)
        print(f"\nDistance to first neighbor:")
        print(f"  Grid distance: {grid_dist} steps")
        print(f"  Geographic distance: {geo_dist:.2f} km")
    
    # Test radius search
    cells_nearby = grid.find_cells_within_radius(test_lat, test_lon, radius_km=10)
    print(f"\nCells within 10km: {len(cells_nearby)}")
    
    # Cache stats
    print(f"\nCache stats: {grid.get_cache_stats()}")
    
    print("\n" + "=" * 80)
