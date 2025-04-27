# tests/test_database.py
import unittest
from unittest.mock import patch, MagicMock
import datetime


# Importer les fonctions à tester
from utils.database import log_interaction, log_feedback, Interaction # Assurez-vous que Interaction est importable


class TestDatabaseFunctions(unittest.TestCase):


    @patch('utils.database.SessionLocal') # Patcher SessionLocal où elle est définie/importée
    def test_log_interaction_success(self, MockSessionLocal):
        """Vérifie que log_interaction appelle les bonnes méthodes de session."""
        # Configurer le mock de la session et de ses méthodes
        mock_db_session = MagicMock()
        MockSessionLocal.return_value = mock_db_session


        # Simuler l'objet Interaction créé et le retour de db.refresh()
        mock_interaction_instance = MagicMock(spec=Interaction)
        mock_interaction_instance.id = 123 # Simuler un ID retourné
        def refresh_side_effect(instance):
            # Simuler l'assignation de l'ID par la DB lors du refresh
            instance.id = 123
        mock_db_session.refresh.side_effect = refresh_side_effect


        # Données d'exemple
        test_query = "Quelle heure est-il ?"
        test_contexts = ["Le contexte 1", "Le contexte 2"]
        test_response = "Il est l'heure de coder !"


        # Appel de la fonction
        returned_id = log_interaction(test_query, test_contexts, test_response)


        # Vérifications
        # 1. Est-ce que SessionLocal() a été appelé pour obtenir une session ?
        MockSessionLocal.assert_called_once()


        # 2. Est-ce que db.add a été appelé avec un objet Interaction ?
        self.assertEqual(mock_db_session.add.call_count, 1)
        added_object = mock_db_session.add.call_args[0][0] # Récupérer l'objet passé à add
        self.assertIsInstance(added_object, Interaction)
        self.assertEqual(added_object.user_query, test_query)
        self.assertEqual(added_object.llm_response, test_response)
        self.assertEqual(added_object.contexts, test_contexts)


        # 3. Est-ce que db.commit a été appelé ?
        mock_db_session.commit.assert_called_once()


        # 4. Est-ce que db.refresh a été appelé (pour obtenir l'ID) ?
        mock_db_session.refresh.assert_called_once_with(added_object)


        # 5. La fonction retourne-t-elle l'ID simulé ?
        self.assertEqual(returned_id, 123)


        # 6. Est-ce que db.close a été appelé ?
        mock_db_session.close.assert_called_once()


    @patch('utils.database.SessionLocal')
    def test_log_feedback_updates_score(self, MockSessionLocal):
      """Vérifie que log_feedback met à jour le score de l'interaction."""
      # Configurer le mock de la session
      mock_db_session = MagicMock()
      MockSessionLocal.return_value = mock_db_session


      # Simuler l'objet Interaction trouvé par la requête
      mock_interaction_found = MagicMock(spec=Interaction)
      mock_interaction_found.id = 456
      mock_interaction_found.feedback_score = None # Score initial


      # Configurer query().filter().first() pour retourner notre mock
      mock_db_session.query.return_value.filter.return_value.first.return_value = mock_interaction_found


      # Données d'exemple
      test_interaction_id = 456
      test_score = 1 # Feedback positif


      # Appel de la fonction
      log_feedback(test_interaction_id, test_score)


      # Vérifications
      # 1. Est-ce que query(Interaction).filter(Interaction.id == ...).first() a été appelé ?
      mock_db_session.query.assert_called_once_with(Interaction)
      mock_db_session.query.return_value.filter.assert_called_once() # Vérifier l'appel à filter
      mock_db_session.query.return_value.filter.return_value.first.assert_called_once()


      # 2. Le score de l'objet Interaction trouvé a-t-il été mis à jour ?
      self.assertEqual(mock_interaction_found.feedback_score, test_score)


      # 3. Est-ce que db.commit a été appelé ?
      mock_db_session.commit.assert_called_once()


      # 4. Est-ce que db.close a été appelé ?
      mock_db_session.close.assert_called_once()




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
