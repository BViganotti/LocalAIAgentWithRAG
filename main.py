#!/usr/bin/env python3
"""
Social Media Content Generation Agent

Transform topics into high-impact social media content for Reddit, LinkedIn, and Twitter
using AI-powered generation with inspiration from successful posts.
"""

from topic_reader import TopicReader
from content_generators import ContentGenerationAgent
from database import get_database
import sys
import os


class ContentAgent:
    """Main content generation agent interface"""
    
    def __init__(self):
        self.topic_reader = TopicReader()
        self.generator = ContentGenerationAgent()
        self.db = get_database()
    
    def show_available_topics(self):
        """Display all available topics"""
        topics = self.topic_reader.get_topic_list()
        
        if not topics:
            print("No topics found in the topic/ directory.")
            print("Please add PDF, TXT, or MD files to the topic/ directory.")
            return
        
        print("\nAvailable Topics:")
        print("-" * 50)
        for i, topic in enumerate(topics, 1):
            print(f"{i}. {topic['title']} ({topic['filename']} - {topic['file_type']})")
        print("-" * 50)
    
    def generate_content_interactive(self):
        """Interactive content generation"""
        while True:
            print("\n" + "="*60)
            print("🤖 Social Media Content Generation Agent")
            print("="*60)
            
            # Show available topics
            self.show_available_topics()
            
            print("\nOptions:")
            print("1. Generate content for a specific topic")
            print("2. Generate content for all topics")
            print("3. View recent generated content")
            print("4. View content statistics")
            print("5. Quit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self.generate_for_specific_topic()
            elif choice == "2":
                self.generate_for_all_topics()
            elif choice == "3":
                self.view_recent_content()
            elif choice == "4":
                self.view_content_stats()
            elif choice == "5":
                print("Goodbye! 🚀")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def generate_for_specific_topic(self):
        """Generate content for a specific topic"""
        topics = self.topic_reader.get_topic_list()
        
        if not topics:
            return
        
        try:
            topic_num = int(input(f"\nEnter topic number (1-{len(topics)}): "))
            if 1 <= topic_num <= len(topics):
                selected_topic = topics[topic_num - 1]
                
                # Load full topic data
                topic_data = self.topic_reader.read_topic_file(
                    os.path.join(self.topic_reader.topic_dir, selected_topic['filename'])
                )
                
                # Choose platforms
                print("\nAvailable platforms:")
                platforms = ['linkedin', 'reddit', 'twitter']
                for i, platform in enumerate(platforms, 1):
                    print(f"{i}. {platform.capitalize()}")
                print("4. All platforms")
                
                platform_choice = input("\nChoose platform (1-4): ").strip()
                
                if platform_choice == "4":
                    selected_platforms = platforms
                elif platform_choice in ["1", "2", "3"]:
                    selected_platforms = [platforms[int(platform_choice) - 1]]
                else:
                    print("Invalid platform choice.")
                    return
                
                print(f"\n🚀 Generating content for {topic_data['title']}...")
                
                results = self.generator.generate_and_save_content(topic_data, selected_platforms)
                
                self.display_generation_results(results)
                
            else:
                print("Invalid topic number.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")
    
    def generate_for_all_topics(self):
        """Generate content for all available topics"""
        topics = self.topic_reader.get_all_topics()
        
        if not topics:
            return
        
        print(f"\n🚀 Generating content for {len(topics)} topics across all platforms...")
        
        total_results = {
            'successful': 0,
            'failed': 0,
            'total_content_pieces': 0
        }
        
        for i, topic_data in enumerate(topics, 1):
            print(f"\nProcessing topic {i}/{len(topics)}: {topic_data['title']}")
            
            try:
                results = self.generator.generate_and_save_content(topic_data)
                total_results['successful'] += 1
                total_results['total_content_pieces'] += len(results['generated_content'])
                
                print(f"✅ Generated {len(results['generated_content'])} pieces of content")
                
                if results['errors']:
                    print(f"⚠️  Errors: {', '.join(results['errors'])}")
                    
            except Exception as e:
                total_results['failed'] += 1
                print(f"❌ Failed: {e}")
        
        print(f"\n📊 Batch Generation Complete:")
        print(f"   Successful topics: {total_results['successful']}")
        print(f"   Failed topics: {total_results['failed']}")
        print(f"   Total content pieces: {total_results['total_content_pieces']}")
    
    def view_recent_content(self):
        """View recently generated content"""
        try:
            if not self.db.connect():
                print("❌ Could not connect to database")
                return
            
            recent_content = self.db.get_recent_content(hours=24, limit=10)
            
            if not recent_content:
                print("No recent content found.")
                return
            
            print(f"\n📝 Recent Content ({len(recent_content)} items):")
            print("-" * 80)
            
            for content in recent_content:
                print(f"Platform: {content['platform'].capitalize()}")
                print(f"Topic: {content['topic_title']}")
                print(f"Created: {content['created_at']}")
                print(f"Content preview: {content['content'][:100]}...")
                print("-" * 80)
                
        except Exception as e:
            print(f"Error viewing content: {e}")
        finally:
            self.db.disconnect()
    
    def view_content_stats(self):
        """View content generation statistics"""
        try:
            if not self.db.connect():
                print("❌ Could not connect to database")
                return
            
            stats = self.db.get_content_stats()
            
            print(f"\n📊 Content Statistics:")
            print(f"Total content pieces: {stats['total_content']}")
            
            if stats['platform_breakdown']:
                print("\nPlatform breakdown:")
                for platform_stat in stats['platform_breakdown']:
                    print(f"  {platform_stat['platform'].capitalize()}: {platform_stat['count']} pieces")
                    
        except Exception as e:
            print(f"Error viewing stats: {e}")
        finally:
            self.db.disconnect()
    
    def display_generation_results(self, results):
        """Display content generation results"""
        print(f"\n📝 Content Generation Results for: {results['topic_title']}")
        print("="*60)
        
        for content in results['generated_content']:
            platform = content['platform'].capitalize()
            char_count = content['metadata']['char_count']
            word_count = content['metadata']['word_count']
            
            print(f"\n🎯 {platform} Content ({word_count} words, {char_count} chars):")
            print("-" * 50)
            print(content['content'])
            print("-" * 50)
        
        if results['saved_ids']:
            print(f"\n💾 Content saved to database:")
            for saved in results['saved_ids']:
                print(f"   {saved['platform'].capitalize()}: ID #{saved['id']}")
        
        # Show generation settings if available
        if 'generation_settings' in results:
            settings = results['generation_settings']
            print(f"\n⚙️  Generation Settings:")
            if settings['content_type'] != 'auto':
                print(f"   Content Type: {settings['content_type'].replace('_', ' ').title()}")
            if settings['tone'] != 'auto':
                print(f"   Tone: {settings['tone'].replace('_', ' ').title()}")
            if settings['num_variations'] > 1:
                print(f"   Variations Generated: {settings['num_variations']}")
        
        if results['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in results['errors']:
                print(f"   {error}")
    
    def advanced_content_generation(self):
        """Advanced content generation with customization options"""
        topics = self.topic_reader.get_topic_list()
        
        if not topics:
            return
        
        try:
            # Select topic
            topic_num = int(input(f"\nEnter topic number (1-{len(topics)}): "))
            if not 1 <= topic_num <= len(topics):
                print("Invalid topic number.")
                return
            
            selected_topic = topics[topic_num - 1]
            topic_data = self.topic_reader.read_topic_file(
                os.path.join(self.topic_reader.topic_dir, selected_topic['filename'])
            )
            
            # Choose platforms
            print("\nAvailable platforms:")
            platforms = ['linkedin', 'reddit', 'twitter']
            for i, platform in enumerate(platforms, 1):
                print(f"{i}. {platform.capitalize()}")
            print("4. All platforms")
            
            platform_choice = input("\nChoose platform (1-4): ").strip()
            
            if platform_choice == "4":
                selected_platforms = platforms
            elif platform_choice in ["1", "2", "3"]:
                selected_platforms = [platforms[int(platform_choice) - 1]]
            else:
                print("Invalid platform choice.")
                return
            
            # Choose content type
            print("\nContent types:")
            content_types = list(ContentType)
            for i, ct in enumerate(content_types, 1):
                print(f"{i}. {ct.value.replace('_', ' ').title()}")
            
            ct_choice = input(f"\nChoose content type (1-{len(content_types)}, or press Enter for auto): ").strip()
            
            if ct_choice:
                try:
                    selected_content_type = content_types[int(ct_choice) - 1]
                except (ValueError, IndexError):
                    selected_content_type = None
            else:
                selected_content_type = None
            
            # Choose tone
            print("\nTone styles:")
            tones = list(ToneStyle)
            for i, tone in enumerate(tones, 1):
                print(f"{i}. {tone.value.replace('_', ' ').title()}")
            
            tone_choice = input(f"\nChoose tone (1-{len(tones)}, or press Enter for auto): ").strip()
            
            if tone_choice:
                try:
                    selected_tone = tones[int(tone_choice) - 1]
                except (ValueError, IndexError):
                    selected_tone = None
            else:
                selected_tone = None
            
            # Choose number of variations
            variations_input = input("\nNumber of variations for A/B testing (1-5, default 1): ").strip()
            try:
                num_variations = int(variations_input) if variations_input else 1
                num_variations = max(1, min(5, num_variations))  # Clamp between 1 and 5
            except ValueError:
                num_variations = 1
            
            print(f"\n🚀 Generating {num_variations} variation(s) for {topic_data['title']}...")
            if selected_content_type:
                print(f"Content Type: {selected_content_type.value.replace('_', ' ').title()}")
            if selected_tone:
                print(f"Tone: {selected_tone.value.replace('_', ' ').title()}")
            
            results = self.generator.generate_and_save_content(
                topic_data, selected_platforms, selected_content_type, selected_tone, num_variations
            )
            
            self.display_generation_results(results)
            
            # Show variations if multiple were generated
            if num_variations > 1:
                self.display_variations(results)
                
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")


    def preview_templates(self):
        """Preview available content templates"""
        print("\n📝 Template Preview System")
        print("=" * 50)
        
        # Choose platform
        platforms = ['linkedin', 'reddit', 'twitter']
        print("\nAvailable platforms:")
        for i, platform in enumerate(platforms, 1):
            print(f"{i}. {platform.capitalize()}")
        
        platform_choice = input("\nChoose platform (1-3): ").strip()
        
        try:
            selected_platform = platforms[int(platform_choice) - 1]
        except (ValueError, IndexError):
            print("Invalid platform choice.")
            return
        
        # Get available content types for platform
        available_types = self.generator.get_available_content_types(selected_platform)
        
        print(f"\nContent types for {selected_platform.capitalize()}:")
        for i, ct in enumerate(available_types, 1):
            print(f"{i}. {ct.value.replace('_', ' ').title()}")
        
        ct_choice = input(f"\nChoose content type (1-{len(available_types)}): ").strip()
        
        try:
            selected_content_type = available_types[int(ct_choice) - 1]
        except (ValueError, IndexError):
            print("Invalid content type choice.")
            return
        
        # Choose tone
        tones = self.generator.get_available_tones()
        print("\nAvailable tones:")
        for i, tone in enumerate(tones, 1):
            print(f"{i}. {tone.value.replace('_', ' ').title()}")
        
        tone_choice = input(f"\nChoose tone (1-{len(tones)}): ").strip()
        
        try:
            selected_tone = tones[int(tone_choice) - 1]
        except (ValueError, IndexError):
            print("Invalid tone choice.")
            return
        
        # Show template preview
        preview = self.generator.preview_template(selected_platform, selected_content_type, selected_tone)
        
        print(f"\n📋 Template Preview: {selected_platform.capitalize()} - {selected_content_type.value.replace('_', ' ').title()} - {selected_tone.value.replace('_', ' ').title()}")
        print("=" * 80)
        print(preview)
        print("=" * 80)
    
    def display_variations(self, results):
        """Display A/B testing variations if available"""
        print("\n🔬 A/B Testing Variations:")
        print("=" * 60)
        
        for content in results['generated_content']:
            if content['metadata'].get('variations') and len(content['metadata']['variations']) > 1:
                platform = content['platform'].capitalize()
                print(f"\n{platform} Variations:")
                print("-" * 40)
                
                variations = content['metadata']['variations']
                for i, variation in enumerate(variations, 1):
                    print(f"\nVariation {i}:")
                    print(f"Length: {len(variation)} chars, {len(variation.split())} words")
                    print(variation[:200] + "..." if len(variation) > 200 else variation)
                    print("-" * 40)


def main():
    """Main entry point"""
    print("🚀 Initializing Enhanced Content Generation Agent...")
    
    # Check if required services are running
    print("📋 Checking prerequisites...")
    
    agent = ContentAgent()
    
    # Check for topics
    topics = agent.topic_reader.get_topic_files()
    if not topics:
        print("\n⚠️  No topic files found!")
        print("Please add PDF, TXT, or MD files to the topic/ directory to get started.")
        print("Example: topic/artificial_intelligence_trends.pdf")
        return
    
    print(f"✅ Found {len(topics)} topic file(s)")
    print("✅ Enhanced content generation agent ready with:")
    print("   • Multi-model support (gpt-oss default)")
    print("   • Content variations & A/B testing")
    print("   • Tone customization")
    print("   • Content templates")
    print("   • Enhanced RAG with hybrid search")
    print("   • Auto-categorization")
    print("   • Dynamic inspiration selection")
    
    # Run interactive mode
    try:
        agent.generate_content_interactive()
    except KeyboardInterrupt:
        print("\n\n👋 Content generation interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()