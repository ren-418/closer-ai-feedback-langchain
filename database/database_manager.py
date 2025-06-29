import os
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import json
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        """Initialize database connection to Supabase."""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    # User Management
    def create_user(self, email: str, password_hash: str, full_name: str) -> Dict:
        """Create a new admin user."""
        try:
            result = self.client.table('users').insert({
                'email': email,
                'password_hash': password_hash,
                'full_name': full_name,
                'role': 'admin'
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[Database] Error creating user: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email."""
        try:
            result = self.client.table('users').select('*').eq('email', email).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[Database] Error getting user: {e}")
            return None
    
    # Closer Management
    def create_closer(self, name: str, email: str = None, phone: str = None, hire_date: str = None) -> Dict:
        """Create a new closer."""
        try:
            closer_data = {'name': name}
            if email:
                closer_data['email'] = email
            if phone:
                closer_data['phone'] = phone
            if hire_date:
                closer_data['hire_date'] = hire_date
            
            result = self.client.table('closers').insert(closer_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[Database] Error creating closer: {e}")
            return None
    
    def get_all_closers(self) -> List[Dict]:
        """Get all active closers."""
        try:
            result = self.client.table('closers').select('*').eq('is_active', True).execute()
            return result.data
        except Exception as e:
            print(f"[Database] Error getting closers: {e}")
            return []
    
    def get_closer_by_id(self, closer_id: str) -> Optional[Dict]:
        """Get closer by ID."""
        try:
            result = self.client.table('closers').select('*').eq('id', closer_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[Database] Error getting closer: {e}")
            return None
    
    # Call Management
    def create_call(self, closer_name: str, transcript_text: str, filename: str = None, 
                   call_date: str = None, closer_id: str = None) -> Dict:
        """Create a new call record."""
        try:
            call_data = {
                'closer_name': closer_name,
                'transcript_text': transcript_text,
                'transcript_length': len(transcript_text),
                'status': 'new'
            }
            
            if filename:
                call_data['filename'] = filename
            if call_date:
                call_data['call_date'] = call_date
            if closer_id:
                call_data['closer_id'] = closer_id
            
            result = self.client.table('calls').insert(call_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[Database] Error creating call: {e}")
            return None
    
    def update_call_analysis(self, call_id: str, analysis_result: Dict) -> bool:
        """Update call with analysis results."""
        try:
            # Extract key metrics from analysis
            final_analysis = analysis_result.get('final_analysis', {})
            executive_summary = final_analysis.get('executive_summary', {})
            metadata = analysis_result.get('metadata', {})
            
            update_data = {
                'status': 'analyzed',
                'total_chunks': metadata.get('total_chunks'),
                'total_reference_files_used': metadata.get('total_reference_files_used'),
                'overall_score': executive_summary.get('overall_score'),
                'letter_grade': executive_summary.get('letter_grade'),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Update call record
            self.client.table('calls').update(update_data).eq('id', call_id).execute()
            
            # Store chunk analyses
            chunk_analyses = analysis_result.get('chunk_analyses', [])
            for chunk in chunk_analyses:
                chunk_data = {
                    'call_id': call_id,
                    'chunk_number': chunk.get('chunk_number'),
                    'total_chunks': chunk.get('total_chunks'),
                    'chunk_text_preview': chunk.get('chunk_text_preview'),
                    'analysis_data': chunk.get('analysis'),
                    'reference_files_used': chunk.get('analysis', {}).get('analysis_metadata', {}).get('reference_files_used', [])
                }
                self.client.table('call_analyses').insert(chunk_data).execute()
            
            # Store final analysis
            final_data = {
                'call_id': call_id,
                'analysis_data': final_analysis,
                'report_metadata': metadata
            }
            self.client.table('final_analyses').insert(final_data).execute()
            
            # Store performance metrics
            performance_data = {
                'call_id': call_id,
                'overall_score': executive_summary.get('overall_score'),
                'letter_grade': executive_summary.get('letter_grade'),
                'created_at': datetime.now().isoformat()
            }
            
            # Extract detailed metrics if available
            detailed_analysis = final_analysis.get('detailed_analysis', {})
            if detailed_analysis:
                performance_data.update({
                    'rapport_building_score': detailed_analysis.get('engagement_rapport', {}).get('score'),
                    'discovery_score': detailed_analysis.get('discovery_qualification', {}).get('score'),
                    'objection_handling_score': detailed_analysis.get('objection_handling', {}).get('score'),
                    'pitch_delivery_score': detailed_analysis.get('pitch_delivery', {}).get('score'),
                    'closing_effectiveness_score': detailed_analysis.get('closing_effectiveness', {}).get('score')
                })
            
            # Extract lead interaction data
            lead_summary = final_analysis.get('lead_interaction_summary', {})
            if lead_summary:
                performance_data.update({
                    'total_objections': lead_summary.get('total_objections_raised'),
                    'total_questions': lead_summary.get('total_questions_asked')
                })
            
            self.client.table('performance_metrics').insert(performance_data).execute()
            
            return True
        except Exception as e:
            print(f"[Database] Error updating call analysis: {e}")
            return False
    
    def get_calls(self, closer_name: str = None, status: str = None, 
                  start_date: str = None, end_date: str = None, limit: int = 100) -> List[Dict]:
        """Get calls with optional filtering."""
        try:
            query = self.client.table('calls').select('*').order('created_at', desc=True).limit(limit)
            
            if closer_name:
                query = query.eq('closer_name', closer_name)
            if status:
                query = query.eq('status', status)
            if start_date:
                query = query.gte('call_date', start_date)
            if end_date:
                query = query.lte('call_date', end_date)
            
            result = query.execute()
            return result.data
        except Exception as e:
            print(f"[Database] Error getting calls: {e}")
            return []
    
    def get_call_by_id(self, call_id: str) -> Optional[Dict]:
        """Get call by ID with all related data."""
        try:
            # Get call data
            call_result = self.client.table('calls').select('*').eq('id', call_id).execute()
            if not call_result.data:
                return None
            
            call_data = call_result.data[0]
            
            # Get chunk analyses
            chunk_result = self.client.table('call_analyses').select('*').eq('call_id', call_id).order('chunk_number').execute()
            call_data['chunk_analyses'] = chunk_result.data
            
            # Get final analysis
            final_result = self.client.table('final_analyses').select('*').eq('call_id', call_id).execute()
            call_data['final_analysis'] = final_result.data[0] if final_result.data else None
            
            return call_data
        except Exception as e:
            print(f"[Database] Error getting call: {e}")
            return None
    
    def update_call_status(self, call_id: str, status: str) -> bool:
        """Update call status."""
        try:
            self.client.table('calls').update({'status': status}).eq('id', call_id).execute()
            return True
        except Exception as e:
            print(f"[Database] Error updating call status: {e}")
            return False
    
    # Analytics
    def get_closer_performance(self, closer_name: str, days: int = 30) -> Dict:
        """Get performance metrics for a specific closer."""
        try:
            # Get calls for the closer
            calls = self.get_calls(closer_name=closer_name)
            
            if not calls:
                return {
                    'closer_name': closer_name,
                    'total_calls': 0,
                    'average_score': 0,
                    'best_score': 0,
                    'worst_score': 0,
                    'grade_distribution': {},
                    'recent_trend': []
                }
            
            # Calculate metrics
            scores = [call.get('overall_score', 0) for call in calls if call.get('overall_score')]
            grades = [call.get('letter_grade') for call in calls if call.get('letter_grade')]
            
            grade_distribution = {}
            for grade in grades:
                grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
            
            return {
                'closer_name': closer_name,
                'total_calls': len(calls),
                'average_score': sum(scores) / len(scores) if scores else 0,
                'best_score': max(scores) if scores else 0,
                'worst_score': min(scores) if scores else 0,
                'grade_distribution': grade_distribution,
                'recent_trend': scores[-10:] if len(scores) > 10 else scores  # Last 10 scores
            }
        except Exception as e:
            print(f"[Database] Error getting closer performance: {e}")
            return {}
    
    def get_team_analytics(self) -> Dict:
        """Get team-wide analytics and leaderboard."""
        try:
            # Get all calls
            calls = self.get_calls(limit=1000)
            
            # Group by closer
            closer_stats = {}
            for call in calls:
                closer_name = call.get('closer_name', 'Unknown')
                score = call.get('overall_score', 0)
                
                if closer_name not in closer_stats:
                    closer_stats[closer_name] = {
                        'name': closer_name,
                        'total_calls': 0,
                        'total_score': 0,
                        'scores': []
                    }
                
                closer_stats[closer_name]['total_calls'] += 1
                closer_stats[closer_name]['total_score'] += score
                closer_stats[closer_name]['scores'].append(score)
            
            # Calculate averages and create leaderboard
            leaderboard = []
            for closer_name, stats in closer_stats.items():
                avg_score = stats['total_score'] / stats['total_calls'] if stats['total_calls'] > 0 else 0
                leaderboard.append({
                    'name': closer_name,
                    'total_calls': stats['total_calls'],
                    'average_score': round(avg_score, 2),
                    'best_score': max(stats['scores']) if stats['scores'] else 0,
                    'recent_trend': stats['scores'][-5:] if len(stats['scores']) > 5 else stats['scores']
                })
            
            # Sort by average score (descending)
            leaderboard.sort(key=lambda x: x['average_score'], reverse=True)
            
            # Calculate team averages
            all_scores = [call.get('overall_score', 0) for call in calls if call.get('overall_score')]
            team_average = sum(all_scores) / len(all_scores) if all_scores else 0
            
            return {
                'team_average': round(team_average, 2),
                'total_calls': len(calls),
                'leaderboard': leaderboard,
                'top_performers': leaderboard[:3] if len(leaderboard) >= 3 else leaderboard
            }
        except Exception as e:
            print(f"[Database] Error getting team analytics: {e}")
            return {}
    
    def search_calls(self, search_term: str, limit: int = 50) -> List[Dict]:
        """Search calls by closer name or filename."""
        try:
            # Search by closer name
            closer_result = self.client.table('calls').select('*').ilike('closer_name', f'%{search_term}%').limit(limit).execute()
            
            # Search by filename
            filename_result = self.client.table('calls').select('*').ilike('filename', f'%{search_term}%').limit(limit).execute()
            
            # Combine and deduplicate
            all_results = closer_result.data + filename_result.data
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                if result['id'] not in seen_ids:
                    seen_ids.add(result['id'])
                    unique_results.append(result)
            
            return unique_results[:limit]
        except Exception as e:
            print(f"[Database] Error searching calls: {e}")
            return []

# Global database manager instance
db_manager = DatabaseManager() 